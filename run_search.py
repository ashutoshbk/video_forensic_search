"""Search an ingested video (run run_ingest.py first). Nothing is embedded again, so every new query only
costs the shortlist + Qwen checks + clip cutting. Several queries can be run in one go (Qwen is loaded once).

    python run_search.py --query query.yaml
    python run_search.py --query q1.yaml q2.yaml
    python run_search.py --text "a person with tattoo" --name tattoo --start 0 --end 300
    python run_search.py --text "blue truck" --target vehicle --reference photos/truck.jpg

Use the same --video / --max-seconds as the ingest (or the same config.yaml) so the right cache is found.
A time and resource log (CPU, RAM, GPU, time per stage and per query) is written to <run folder>/logs/.
"""
import argparse
import gc
import os
import sys
import time
import traceback

import torch

from forensic.clips import cut_all
from forensic.config import load_config, load_query, mmss, run_dirs
from forensic.decide import decide
from forensic.embed import ClipEncoder
from forensic.monitor import RunMonitor
from forensic.reference import load_reference
from forensic.search import IngestCache, shortlist
from forensic.verify import Qwen, check_tracks, plan_and_save


SKIP_MSGS = ("No detections of this target", "All detections were filtered", "No track has enough")


def run_one(args, cfg_path, video, shared):
    """One video: its own config and cache; CLIP and Qwen come from `shared` and are loaded only once per process."""
    cfg = load_config(cfg_path, {"video": video, "max_seconds": args.max_seconds})
    dirs = run_dirs(cfg)

    queries = [load_query(p) for p in (args.query or [])]
    if args.text:
        queries.append(load_query(None, dict(text=args.text, name=args.name, target=args.target, search_start=args.start,
                                             search_end=args.end, reference=args.reference)))
    if not queries:
        queries = [load_query("query.yaml")]

    with RunMonitor("search", os.path.join(dirs["out"], "logs"), cfg=cfg,
                    extra=dict(queries=", ".join(f"{q['name']} ({q['text']})" for q in queries))) as mon:
        with mon.stage("load_ingest_cache"):
            cache = IngestCache(cfg, dirs)
        if shared["clip"] is None:
            with mon.stage("load_clip_model"):
                shared["clip"] = ClipEncoder(cfg["clip_arch"], cfg["clip_weights"], cfg["device"], cfg["emb_batch"], cfg["num_workers"])
        clip = shared["clip"]

        for q in queries:
            name = q["name"]
            fl = cfg.get("abs_floor_" + q["target"]) or cfg.get("abs_floor")
            stats = shared["stats"].setdefault(name, dict(
                query=q["text"], videos=0, dur=0.0, extracted=0, sent=0, qwen=0.0, match=0,
                settings=f"z_min={cfg.get('z_min')} min_tracks={cfg.get('min_tracks')} z_pad={cfg.get('z_pad')} "
                         f"topk={cfg.get('topk')} abs_floor={fl} n_crops={cfg.get('n_crops')}"))
            stats["videos"] += 1
            stats["dur"] += cache.nfr / cache.fps
            print(f"\n{'=' * 70}\nQUERY [{name}]: {q['text']}  (target: {q['target']})\n{'=' * 70}")
            resq = os.path.join(dirs["results"], name)
            os.makedirs(resq, exist_ok=True)

            with mon.stage("reference_photo", name):
                ref_pil, ref_vec = load_reference(q["reference"], clip, q["require_reference"])

            if shared["qwen"] is None:
                with mon.stage("load_qwen"):
                    shared["qwen"] = Qwen(cfg)
            qwen = shared["qwen"]

            with mon.stage("plan_query", name):
                attr, same_q = plan_and_save(qwen, q, resq)

            sl = None
            with mon.stage("shortlist", name) as st:
                try:
                    sl = shortlist(cache, cfg, q, clip, ref_pil, ref_vec, resq)
                    st.note(tracks_shortlisted=len(sl["TOP"]))
                except RuntimeError as e:
                    if not str(e).startswith(SKIP_MSGS):
                        raise                                 # a real error: keep failing loudly
                    print(f"[skip] {name}: {e}", flush=True)
            if sl is not None:
                stats["extracted"] += len(sl["groups"])
                stats["sent"] += len(sl["TOP"])
            if sl is None or not sl["TOP"]:
                print("[skip] nothing to check in this video: Qwen, decision and clips skipped", flush=True)
                print(f"[time] QUERY TOTAL [{name}]: {mmss(mon.group_seconds(name))}. Results in: {resq}")
                continue

            with mon.stage("qwen_checks", name) as st:
                _t0 = time.time()
                QRES = check_tracks(qwen, cache, cfg, sl["TOP"], attr, same_q, ref_pil)
                stats["qwen"] += time.time() - _t0
                st.note(tracks=len(sl["TOP"]), crops=sum(len(t["crops"]) for t in sl["TOP"]))

            with mon.stage("decision", name) as st:
                final = decide(cache, cfg, q, resq, sl["TOP"], QRES)
                stats["match"] += sum(x["verdict"] == "MATCH" for x in final)
                st.note(match=sum(x["verdict"] == "MATCH" for x in final),
                        review=sum(x["verdict"].startswith("REVIEW") for x in final))

            with mon.stage("clips", name):
                cut_all(cache, cfg, q, resq, sl["TOP"], QRES, final)

            print(f"[time] QUERY TOTAL [{name}]: {mmss(mon.group_seconds(name))}. Results in: {resq}")


def hhmm(sec):
    m = int(round(sec / 60))
    return f"{m // 60}:{m % 60:02d}"


def append_run_log(shared, wall):
    """One row per query in run_log.xlsx (created on the first run, appended after that)."""
    if not shared["stats"]:
        return
    try:
        from openpyxl import Workbook, load_workbook
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_log.xlsx")
        if os.path.isfile(path):
            wb = load_workbook(path)
            ws = wb.active
        else:
            wb = Workbook()
            ws = wb.active
            ws.title = "runs"
            ws.append(["Sl no", "Date", "Run name", "Query", "Videos targeted", "Total video duration (h:mm)",
                       "Tracks extracted (after filters)", "Tracks sent to Qwen", "Qwen check time (s)",
                       "Matches", "Total run time (s)", "Settings"])
            for col, w in zip("ABCDEFGHIJKL", (7, 17, 14, 40, 10, 14, 16, 12, 14, 9, 12, 60)):
                ws.column_dimensions[col].width = w
            ws.freeze_panes = "A2"
        for name, st in shared["stats"].items():
            ws.append([ws.max_row, time.strftime("%Y-%m-%d %H:%M"), name, st["query"], st["videos"],
                       hhmm(st["dur"]), st["extracted"], st["sent"], round(st["qwen"]),
                       st["match"], round(wall), st["settings"]])
        wb.save(path)
        print(f"[runlog] {len(shared['stats'])} row(s) added to {path}", flush=True)
    except Exception as e:
        print(f"[runlog] could not write run_log.xlsx ({e}). Close it in Excel and re-run.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", help="overrides `video` in the config")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("CONFIG", "VIDEO"),
                    help="repeat for several videos: one process, CLIP and Qwen loaded once")
    ap.add_argument("--max-seconds", type=float, help="overrides `max_seconds` in the config")
    ap.add_argument("--query", nargs="+", help="one or more query files (see query.yaml)")
    ap.add_argument("--text", help="a query written on the command line instead of a file")
    ap.add_argument("--name", help="results folder name for --text")
    ap.add_argument("--target", choices=["person", "vehicle"])
    ap.add_argument("--start", type=float, help="search from this video second")
    ap.add_argument("--end", type=float, help="search until this video second")
    ap.add_argument("--reference", help="path of a reference photo")
    args = ap.parse_args()

    runs = [(c, v) for c, v in args.pair] if args.pair else [(args.config, args.video)]
    shared = dict(clip=None, qwen=None, stats={})
    t_start = time.time()
    failed = 0
    for i, (cfg_path, video) in enumerate(runs):
        print(f"[video-start] {i}", flush=True)
        try:
            run_one(args, cfg_path, video, shared)
            print(f"[video-done] {i}", flush=True)
        except Exception as e:
            failed += 1
            traceback.print_exc()
            print(f"[video-failed] {i}: {e}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    append_run_log(shared, time.time() - t_start)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
