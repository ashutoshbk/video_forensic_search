"""Forensic Search UI server.

Runs on the GPU machine, inside the video_forensic_search repo folder.
It does three things:
  1. serves the web page (static/index.html)
  2. starts run_search.py for a typed prompt (one job at a time) and reports progress
  3. reads the result files that run_search.py already writes and returns them ranked by score

Start it from the repo folder:
    python -m uvicorn ui_app2.server:app --host 0.0.0.0 --port 8000
"""
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import queue
import uuid
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# ---------------------------------------------------------------- settings
REPO = Path(os.environ.get("FORENSIC_REPO", Path(__file__).resolve().parent.parent))
OUT = REPO / "outputs"
LOGS = OUT / "_ui_logs"
STATIC = Path(__file__).resolve().parent / "static"

# folders that hold the source videos (used to find each ingested video)
VIDEO_DIRS = [Path(p) for p in os.environ.get(
    "FORENSIC_VIDEO_DIRS",
    "/mnt/storage/ashutosh_stuff/recording:/mnt/storage/Arka/forensic_test_videos:/mnt/storage/Arka",
).split(":")]
# videos in this folder are the CCTV cameras: searched with config_indoor.yaml and
# grouped under "All cameras"
INDOOR_DIR = Path(os.environ.get("FORENSIC_INDOOR_DIR", "/mnt/storage/ashutosh_stuff/recording"))
INDOOR_CONFIG = "config_indoor.yaml"

VEHICLE_WORDS = re.compile(
    r"\b(car|cars|truck|van|bus|vehicle|motorcycle|motorbike|lorry|taxi|suv|number ?plate|numberplate)\b", re.I)
SAFE = re.compile(r"[A-Za-z0-9_.\-]+")

app = FastAPI(title="Forensic Search UI")


# ---------------------------------------------------------------- helpers
def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv(p):
    try:
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def find_video(stem):
    for d in VIDEO_DIRS:
        p = d / f"{stem}.mp4"
        if p.is_file():
            return p
    return None


def is_indoor(video):
    try:
        return video.resolve().parent == INDOOR_DIR.resolve()
    except Exception:
        return False


def list_cameras():
    """Every ingested video whose source file can be found."""
    cams = []
    if not OUT.is_dir():
        return cams
    for d in sorted(OUT.iterdir()):
        if d.is_dir() and d.name.endswith("_full") and any(d.iterdir()):
            stem = d.name[: -len("_full")]
            v = find_video(stem)
            if v:
                cams.append({"id": stem, "label": stem, "indoor": is_indoor(v)})
    return cams


def url_for(p):
    return "/files/" + quote(p.relative_to(OUT).as_posix())


def find_result_dir(stem, name):
    if not (SAFE.fullmatch(stem) and SAFE.fullmatch(name)):
        return None
    return next(OUT.glob(f"{stem}_*/results/{name}"), None)


FB_FILE = OUT / "_feedback" / "feedback.json"
_fb_lock = threading.Lock()


def fb_load():
    d = read_json(FB_FILE)
    return d if isinstance(d, dict) else {}


def fb_key(video, name, clip):
    return f"{video}/{name}/{clip}"


def load_clips(rdir, stem):
    """Confirmed + review clips of one search, with the scores from final_results.json."""
    final = read_json(rdir / "final_results.json")
    by_track = {str(r.get("track")): r for r in final} if isinstance(final, list) else {}
    fb = fb_load()
    clips = []
    for sub, status in (("clips", "confirmed"), ("review_clips", "review")):
        for row in read_csv(rdir / sub / "clips_index.csv"):
            f = rdir / sub / (row.get("clip") or "")
            if not f.is_file():
                continue
            bt = str(row.get("best_track", ""))
            ev = by_track.get(bt, {})
            try:
                score = float(row.get("best_score") or 0)
            except ValueError:
                score = 0.0
            clips.append({
                "video": stem,
                "name": rdir.name,
                "feedback": fb.get(fb_key(stem, rdir.name, row.get("clip")), {}).get("verdict", ""),
                "status": status,
                "clip": row.get("clip"),
                "url": url_for(f),
                "start": row.get("start"),
                "end": row.get("end"),
                "seconds": row.get("seconds"),
                "tracks": row.get("tracks"),
                "best_track": bt,
                "score": score,
                "reason": row.get("reason"),
                "evidence": {k: ev.get(k) for k in
                             ("strict", "raw", "avg", "clip", "final", "px", "verdict", "best_t", "first", "last")},
            })
    clips.sort(key=lambda c: c["score"], reverse=True)
    return clips


def tail(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4000))
            txt = f.read().decode("utf-8", "ignore")
        parts = [p.strip() for p in re.split(r"[\r\n]+", txt) if p.strip()]
        return parts[-1][:200] if parts else ""
    except Exception:
        return ""


# Browsers only play H.264 mp4. Clips cut with OpenCV are often MPEG-4 part 2 and show
# "No video with supported format", so an H.264 copy is made once and cached in outputs/_web/.
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe")
WEB = OUT / "_web"
_web_lock = threading.Lock()


def codec_of(p):
    if not FFPROBE:
        return None
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip().split(",")[0] or None
    except Exception:
        return None


def web_version(p):
    if p.suffix.lower() != ".mp4" or not FFMPEG:
        return p
    dst = WEB / p.relative_to(OUT)
    if dst.is_file():
        return dst
    with _web_lock:
        if dst.is_file():
            return dst
        if codec_of(p) == "h264":
            return p
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".tmp.mp4")
        r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(p), "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-an",
                            "-movflags", "+faststart", str(tmp)], capture_output=True, text=True)
        if r.returncode != 0 or not tmp.is_file():
            print("ffmpeg failed:", r.stderr[-300:])
            return p
        tmp.replace(dst)
        return dst


# ---------------------------------------------------------------- jobs
JOBS = {}
QUEUE = queue.Queue()


def public(job):
    fb = fb_load()
    results = [dict(c, feedback=fb.get(fb_key(c["video"], c["name"], c["clip"]), {}).get("verdict", ""))
               for c in job["results"]]
    return {k: v for k, v in job.items() if k not in ("proc", "cancel")} | {"results": results} | {
        "elapsed": round((time.time() - job["started"]) if job.get("started") and job["status"] == "running"
                         else job.get("elapsed_final", 0), 1)}


def run_job(job):
    import re as _re
    job["status"] = "running"
    job["started"] = time.time()
    LOGS.mkdir(parents=True, exist_ok=True)
    items = []
    for stem in job["videos"]:
        video = find_video(stem)
        if not video:
            job["errors"].append(f"{stem}: source video not found")
            continue
        items.append((stem, video))

    def add_results(stem):
        try:
            _add_results(stem)
        except Exception as e:  # a skipped video has no clips
            job["log"] = f"{stem}: no clips to load ({e})"

    def _add_results(stem):
        rdir = find_result_dir(stem, job["name"])
        if rdir:
            new = load_clips(rdir, stem)
            for c in new:  # make browser-playable copies now so the page opens instantly
                try:
                    web_version(OUT / unquote(c["url"][len("/files/"):]))
                except Exception:
                    pass
            job["results"] += new
            job["results"].sort(key=lambda c: c["score"], reverse=True)

    if items and not job["cancel"]:
        cmd = [sys.executable, "run_search.py"]
        for stem, video in items:
            cmd += ["--pair", INDOOR_CONFIG if is_indoor(video) else "config.yaml", str(video)]
        cmd += ["--text", job["text"], "--name", job["name"], "--target", job["target"],
                "--start", "0", "--end", "999999"]
        logf = LOGS / f"{job['id']}_all.log"
        done, bad = set(), set()

        def collect():
            try:
                text = logf.read_text(errors="ignore")
            except OSError:
                return
            starts = [int(x) for x in _re.findall(r"\[video-start\] (\d+)", text)]
            if starts:
                job["index"], job["current"] = starts[-1], items[starts[-1]][0]
            for x in _re.findall(r"\[video-done\] (\d+)", text):
                i = int(x)
                if i not in done:
                    done.add(i)
                    add_results(items[i][0])
            for x in _re.findall(r"\[video-failed\] (\d+)", text):
                i = int(x)
                if i not in bad:
                    bad.add(i)
                    job["errors"].append(f"{items[i][0]}: search failed, see outputs/_ui_logs/{logf.name}")

        with open(logf, "w") as lf:
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            job["proc"] = proc
            while proc.poll() is None:
                time.sleep(1)
                job["log"] = tail(logf)
                collect()
        job["log"] = tail(logf)
        collect()
        if proc.returncode != 0 and not job["cancel"] and not bad:
            job["errors"].append(f"search failed, see outputs/_ui_logs/{logf.name}")
    job["elapsed_final"] = time.time() - job["started"]
    job["status"] = "cancelled" if job["cancel"] else "done"
    job["index"] = len(job["videos"])


def worker():
    while True:
        job = JOBS[QUEUE.get()]
        if job["status"] == "queued":
            try:
                run_job(job)
            except Exception as e:  # keep the worker alive
                job["errors"].append(f"internal error: {e}")
                job["status"] = "done"


threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- API
class SearchReq(BaseModel):
    text: str
    cameras: list[str] = ["all"]
    target: str = "auto"  # auto | person | vehicle


@app.get("/api/cameras")
def api_cameras():
    return list_cameras()


@app.post("/api/search")
def api_search(req: SearchReq):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Type a prompt first")
    cams = list_cameras()
    ids = {c["id"] for c in cams}
    if "all" in req.cameras:
        videos = [c["id"] for c in cams if c["indoor"]]
    else:
        videos = [c for c in req.cameras if c in ids]
    if not videos:
        raise HTTPException(400, "No ingested video selected")
    target = req.target if req.target in ("person", "vehicle") else \
        ("vehicle" if VEHICLE_WORDS.search(text) else "person")
    jid = uuid.uuid4().hex[:8]
    job = {"id": jid, "text": text, "videos": videos, "target": target,
           "name": "ui_" + time.strftime("%Y%m%d_%H%M%S"), "status": "queued", "cancel": False,
           "current": None, "index": 0, "log": "", "results": [], "errors": [], "started": None}
    JOBS[jid] = job
    QUEUE.put(jid)
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def api_job(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "Unknown job")
    return public(job)


@app.post("/api/jobs/{jid}/cancel")
def api_cancel(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "Unknown job")
    job["cancel"] = True
    if job["status"] == "queued":
        job["status"] = "cancelled"
    proc = job.get("proc")
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
    return {"ok": True}


@app.get("/api/history")
def api_history():
    items = []
    for p in OUT.glob("*_full/results/*/plan.json"):
        rdir = p.parent
        stem = rdir.parents[1].name[: -len("_full")]
        plan = read_json(p) or {}
        items.append({
            "video": stem, "name": rdir.name, "query": plan.get("pos") or rdir.name,
            "time": p.stat().st_mtime,
            "confirmed": len(read_csv(rdir / "clips" / "clips_index.csv")),
            "review": len(read_csv(rdir / "review_clips" / "clips_index.csv")),
        })
    items.sort(key=lambda x: x["time"], reverse=True)
    return items[:200]


@app.get("/api/result")
def api_result(video: str, name: str):
    rdir = find_result_dir(video, name)
    if not rdir:
        raise HTTPException(404, "No such result")
    plan = read_json(rdir / "plan.json") or {}
    return {"query": plan.get("pos") or name, "video": video, "results": load_clips(rdir, video)}


class FeedbackReq(BaseModel):
    video: str
    name: str
    clip: str
    verdict: str  # "correct" | "incorrect" | "" (clears the mark)


@app.post("/api/feedback")
def api_feedback(req: FeedbackReq):
    if req.verdict not in ("correct", "incorrect", ""):
        raise HTTPException(400, "verdict must be correct or incorrect")
    rdir = find_result_dir(req.video, req.name)
    if not rdir:
        raise HTTPException(404, "No such result")
    c = next((x for x in load_clips(rdir, req.video) if x["clip"] == req.clip), None)
    if not c:
        raise HTTPException(404, "No such clip")
    plan = read_json(rdir / "plan.json") or {}
    key = fb_key(req.video, req.name, req.clip)
    with _fb_lock:
        fb = fb_load()
        if req.verdict:
            fb[key] = {"verdict": req.verdict, "query": plan.get("pos") or req.name, "video": req.video,
                       "name": req.name, "clip": req.clip, "status": c["status"], "score": c["score"],
                       "track": c["best_track"], "start": c["start"], "end": c["end"],
                       "marked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        else:
            fb.pop(key, None)
        FB_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = FB_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(fb, indent=1), encoding="utf-8")
        tmp.replace(FB_FILE)
    return {"ok": True}


@app.get("/api/feedback")
def api_feedback_list():
    return sorted(fb_load().values(), key=lambda x: x.get("marked_at", ""), reverse=True)


@app.get("/api/feedback.csv")
def api_feedback_csv():
    cols = ["marked_at", "verdict", "query", "video", "name", "clip", "status", "score", "track", "start", "end"]
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in api_feedback_list():
        w.writerow(r)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=feedback.csv"})


@app.get("/files/{path:path}")
def files(path: str):
    p = (OUT / path).resolve()
    if OUT.resolve() not in p.parents or not p.is_file():
        raise HTTPException(404)
    if p.suffix.lower() == ".mp4":
        return FileResponse(web_version(p), media_type="video/mp4")
    return FileResponse(p)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
