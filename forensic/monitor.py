"""Run monitor: records how long every stage takes and how many resources it uses.

    with RunMonitor("ingest", log_dir, cfg=cfg) as mon:
        with mon.stage("detect_track_crop") as st:
            ...
            st.note(detections=1234, video_seconds=3600)

For every stage it logs: duration, CPU %, RAM, GPU utilisation and GPU memory (all sampled every few seconds in a
background thread), plus the peak GPU memory PyTorch allocated. It also logs the machine (CPU, RAM, GPUs, versions),
the settings of the run and the git commit. Three files are written to `log_dir`, also when the run fails or is
stopped with Ctrl+C:
    run_<script>_<time>.txt          readable summary table  (send this one)
    run_<script>_<time>.json         everything, machine-readable
    run_<script>_<time>_samples.csv  the raw samples over time (for graphs)

Optional packages: psutil (CPU/RAM) and nvidia-ml-py (GPU; without it `nvidia-smi` is used if it exists).
Whatever is missing is simply left out of the log.
"""
import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime

try:
    import psutil
except ImportError:                       # CPU / RAM columns are then left out
    psutil = None

GIB = 2 ** 30


def fmt_s(s):
    s = float(s)
    if s < 90:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


class _GpuReader:
    """GPU utilisation / memory per device through NVML, or `nvidia-smi` as a fallback."""

    def __init__(self):
        self.nvml, self.handles, self.smi, self.devices = None, [], False, []
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                name = name.decode() if isinstance(name, bytes) else name
                self.handles.append(h)
                self.devices.append(dict(index=i, name=name, total_gib=round(pynvml.nvmlDeviceGetMemoryInfo(h).total / GIB, 1)))
        except Exception:
            self.nvml, self.handles, self.devices = None, [], []
        if not self.devices and shutil.which("nvidia-smi"):
            try:
                out = self._smi("index,name,memory.total")
                for line in out:
                    i, name, tot = [x.strip() for x in line.split(",")]
                    self.devices.append(dict(index=int(i), name=name, total_gib=round(float(tot) / 1024, 1)))
                self.smi = bool(self.devices)
            except Exception:
                self.devices, self.smi = [], False

    @staticmethod
    def _smi(fields):
        res = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return [l for l in res.stdout.strip().splitlines() if l.strip()]

    def read(self):
        """One dict per GPU: util (%), mem_gib, power_w (None if not available)."""
        out = []
        try:
            if self.nvml:
                for h in self.handles:
                    u = self.nvml.nvmlDeviceGetUtilizationRates(h).gpu
                    m = self.nvml.nvmlDeviceGetMemoryInfo(h).used / GIB
                    try:
                        p = self.nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    except Exception:
                        p = None
                    out.append(dict(util=float(u), mem_gib=m, power_w=p))
            elif self.smi:
                for line in self._smi("utilization.gpu,memory.used,power.draw"):
                    u, m, p = [x.strip() for x in line.split(",")]
                    out.append(dict(util=float(u), mem_gib=float(m) / 1024,
                                    power_w=float(p) if p.replace(".", "", 1).isdigit() else None))
        except Exception:
            return []
        return out


class _Stage:
    def __init__(self):
        self.notes = {}

    def note(self, **kw):
        """Attach numbers to the stage (e.g. detections=..., video_seconds=...)."""
        self.notes.update(kw)


class RunMonitor:
    def __init__(self, script, log_dir, cfg=None, extra=None, interval=2.0):
        self.script, self.log_dir, self.cfg, self.extra, self.interval = script, log_dir, cfg or {}, extra or {}, interval
        self.stages, self.samples = [], []
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread = None
        self.gpu = _GpuReader()
        self.proc = psutil.Process() if psutil else None
        self.status = "ok"
        self.t_start_wall = None

    # ---------------- sampling ----------------
    def _sample(self):
        s = dict(t=time.time())
        if psutil:
            vm = psutil.virtual_memory()
            s.update(cpu=psutil.cpu_percent(interval=None), ram_gib=vm.used / GIB, ram_pct=vm.percent)
            try:
                rss = self.proc.memory_info().rss
                for c in self.proc.children(recursive=True):       # data-loader workers etc.
                    try:
                        rss += c.memory_info().rss
                    except Exception:
                        pass
                s["proc_gib"] = rss / GIB
            except Exception:
                pass
        s["gpus"] = self.gpu.read()
        with self._lock:
            self.samples.append(s)
        return s

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    # ---------------- run ----------------
    def __enter__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.t_start_wall = time.time()
        self.t_start = time.perf_counter()
        self.started = datetime.now()
        if psutil:
            psutil.cpu_percent(interval=None)                        # the first call always returns 0.0
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.status = "ok"
        elif issubclass(exc_type, KeyboardInterrupt):
            self.status = "interrupted (Ctrl+C)"
        else:
            self.status = f"failed: {exc_type.__name__}: {exc}"
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._sample()
        self.total_s = time.perf_counter() - self.t_start
        try:
            self._write()
        except Exception as e:                                        # a logging problem must never hide the real error
            print(f"[monitor] could not write the log: {e}")
        return False

    @contextmanager
    def stage(self, name, group=None):
        """Time one stage and measure its resources. Re-raises errors after recording them."""
        st = _Stage()
        torch = None
        try:
            import torch as _t
            if _t.cuda.is_available():
                torch = _t
                for i in range(_t.cuda.device_count()):
                    _t.cuda.reset_peak_memory_stats(i)
        except Exception:
            torch = None
        t0_wall, t0 = time.time(), time.perf_counter()
        self._sample()
        status = "ok"
        try:
            yield st
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except BaseException:
            status = "failed"
            raise
        finally:
            dur = time.perf_counter() - t0
            self._sample()
            rec = self._stats(t0_wall, time.time())
            rec.update(name=name, group=group, status=status, seconds=round(dur, 2),
                       started=datetime.fromtimestamp(t0_wall).isoformat(timespec="seconds"), notes=dict(st.notes))
            if torch is not None:
                try:
                    rec["torch_vram_peak_gib"] = [round(torch.cuda.max_memory_allocated(i) / GIB, 2)
                                                  for i in range(torch.cuda.device_count())]
                except Exception:
                    pass
            vs = rec["notes"].get("video_seconds")
            if vs and dur > 0:
                rec["notes"]["x_realtime"] = round(vs / dur, 1)
            self.stages.append(rec)
            print(f"[time] {(group + ' / ') if group else ''}{name}: {fmt_s(dur)}")

    def group_seconds(self, group):
        return sum(s["seconds"] for s in self.stages if s["group"] == group)

    # ---------------- statistics ----------------
    def _stats(self, t0, t1):
        with self._lock:
            ss = [s for s in self.samples if t0 - 1e-6 <= s["t"] <= t1 + 1e-6]
        out = {}
        avg = lambda v: round(sum(v) / len(v), 1) if v else None
        mx = lambda v: round(max(v), 1) if v else None
        cpu = [s["cpu"] for s in ss if "cpu" in s]
        out["cpu_avg"], out["cpu_peak"] = avg(cpu), mx(cpu)
        ram = [s["ram_gib"] for s in ss if "ram_gib" in s]
        out["ram_peak_gib"] = round(max(ram), 2) if ram else None
        pr = [s["proc_gib"] for s in ss if "proc_gib" in s]
        out["proc_ram_peak_gib"] = round(max(pr), 2) if pr else None
        gpus = []
        for i in range(len(self.gpu.devices)):
            u = [s["gpus"][i]["util"] for s in ss if len(s["gpus"]) > i]
            m = [s["gpus"][i]["mem_gib"] for s in ss if len(s["gpus"]) > i]
            p = [s["gpus"][i]["power_w"] for s in ss if len(s["gpus"]) > i and s["gpus"][i]["power_w"] is not None]
            gpus.append(dict(index=i, util_avg=avg(u), util_peak=mx(u), mem_peak_gib=round(max(m), 2) if m else None,
                             power_avg_w=avg(p)))
        out["gpus"] = gpus
        out["n_samples"] = len(ss)
        return out

    # ---------------- output ----------------
    def _machine(self):
        info = dict(host=socket.gethostname(), os=platform.platform(), python=sys.version.split()[0],
                    cpu=platform.processor() or platform.machine(), gpus=self.gpu.devices)
        if psutil:
            info.update(cpu_cores_logical=psutil.cpu_count(), cpu_cores_physical=psutil.cpu_count(logical=False),
                        ram_total_gib=round(psutil.virtual_memory().total / GIB, 1))
        try:
            import torch
            info["torch"] = torch.__version__
            info["cuda"] = torch.version.cuda
        except Exception:
            pass
        for mod in ("ultralytics", "open_clip", "transformers"):
            try:
                info[mod] = __import__(mod).__version__
            except Exception:
                pass
        try:
            r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
            info["git_commit"] = r.stdout.strip() or None
        except Exception:
            info["git_commit"] = None
        if not self.gpu.devices:
            info["gpu_note"] = "no GPU metrics available (no NVIDIA driver / pynvml / nvidia-smi found)"
        return info

    def _write(self):
        stamp = self.started.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.log_dir, f"run_{self.script}_{stamp}")
        machine = self._machine()
        report = dict(script=self.script, started=self.started.isoformat(timespec="seconds"), status=self.status,
                      total_seconds=round(self.total_s, 2), machine=machine, settings=self.cfg, extra=self.extra,
                      stages=self.stages)
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
        with open(base + "_samples.csv", "w", newline="") as f:
            w = csv.writer(f)
            ng = len(self.gpu.devices)
            w.writerow(["time", "seconds_from_start", "cpu_pct", "ram_used_gib", "process_ram_gib"] +
                       [c for i in range(ng) for c in (f"gpu{i}_util_pct", f"gpu{i}_mem_gib")])
            for s in self.samples:
                row = [datetime.fromtimestamp(s["t"]).isoformat(timespec="seconds"), round(s["t"] - self.t_start_wall, 1),
                       s.get("cpu", ""), round(s["ram_gib"], 2) if "ram_gib" in s else "",
                       round(s["proc_gib"], 2) if "proc_gib" in s else ""]
                for i in range(ng):
                    g = s["gpus"][i] if len(s["gpus"]) > i else None
                    row += [g["util"], round(g["mem_gib"], 2)] if g else ["", ""]
                w.writerow(row)
        text = self._text(report)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("\n" + text)
        print(f"[monitor] log files: {base}.txt / .json / _samples.csv")

    def _text(self, rep):
        m = rep["machine"]
        L = [f"RUN LOG - {rep['script']} - started {rep['started']} - status: {rep['status']}",
             f"TOTAL TIME: {fmt_s(rep['total_seconds'])}", "",
             f"Machine : {m['host']} | {m['os']} | python {m['python']}"]
        if "cpu_cores_logical" in m:
            L.append(f"CPU/RAM : {m['cpu']} | {m['cpu_cores_physical']} cores ({m['cpu_cores_logical']} threads) | {m['ram_total_gib']} GiB RAM")
        gp = "; ".join(f"gpu{g['index']} {g['name']} {g['total_gib']} GiB" for g in m["gpus"]) or m.get("gpu_note", "no GPU")
        L.append(f"GPUs    : {gp}")
        L.append("Versions: " + ", ".join(f"{k} {m[k]}" for k in ("torch", "cuda", "ultralytics", "open_clip", "transformers") if k in m)
                 + f" | git commit {m.get('git_commit')}")
        for k in ("video", "max_seconds", "stride", "topk", "n_crops"):
            if k in rep["settings"]:
                L.append(f"{k:<8}: {rep['settings'][k]}")
        for k, v in rep["extra"].items():
            L.append(f"{k:<8}: {v}")
        L += ["", "STAGES (CPU and RAM are for the whole machine; GPU memory is what the GPU driver reports)", ""]
        for s in rep["stages"]:
            L.append(f"{(s['group'] + ' / ') if s['group'] else ''}{s['name']}   [{s['status']}]   {fmt_s(s['seconds'])}")
            res = []
            if s["cpu_avg"] is not None:
                res.append(f"CPU avg {s['cpu_avg']}% / peak {s['cpu_peak']}%")
            if s["ram_peak_gib"] is not None:
                res.append(f"RAM peak {s['ram_peak_gib']} GiB (this program {s['proc_ram_peak_gib']} GiB)")
            L.append("    " + " | ".join(res) if res else "    (no CPU/RAM data: install psutil)")
            for g in s["gpus"]:
                extra = f" | power avg {g['power_avg_w']} W" if g["power_avg_w"] is not None else ""
                L.append(f"    gpu{g['index']}: util avg {g['util_avg']}% / peak {g['util_peak']}% | memory peak {g['mem_peak_gib']} GiB{extra}")
            if "torch_vram_peak_gib" in s:
                L.append(f"    PyTorch VRAM peak per GPU (GiB): {s['torch_vram_peak_gib']}")
            if s["notes"]:
                L.append("    " + ", ".join(f"{k}={v}" for k, v in s["notes"].items()))
        return "\n".join(L) + "\n"
