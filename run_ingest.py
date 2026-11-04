"""Ingest one video: detect + track people and vehicles, save crops, compute OpenCLIP vectors.
Run once per video (resumable), then run run_search.py for every query.

    python run_ingest.py                       # uses config.yaml
    python run_ingest.py --video clip.mp4 --max-seconds 300

A time and resource log (CPU, RAM, GPU, time per stage) is written to <run folder>/logs/.
"""
import argparse
import os

from forensic.config import load_config, mmss, run_dirs
from forensic.embed import ClipEncoder, embed_all_crops
from forensic.ingest import run_ingest, video_info
from forensic.monitor import RunMonitor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", help="overrides `video` in the config")
    ap.add_argument("--max-seconds", type=float, help="overrides `max_seconds` in the config")
    args = ap.parse_args()

    cfg = load_config(args.config, {"video": args.video, "max_seconds": args.max_seconds})
    dirs = run_dirs(cfg)
    had_state = os.path.exists(os.path.join(dirs["out"], "state.json"))   # a resumed / finished ingest is not a full timing

    with RunMonitor("ingest", os.path.join(dirs["out"], "logs"), cfg=cfg) as mon:
        with mon.stage("video_info"):
            info = video_info(cfg, dirs)

        with mon.stage("detect_track_crop") as st:
            rows = run_ingest(cfg, dirs, info)
            st.note(detections=len(rows), tracks=len({r["track"] for r in rows}))
            if had_state:
                st.note(resumed_or_cached="yes (time covers only the new part)")
            else:
                st.note(video_seconds=round(info["used_s"], 1))

        with mon.stage("load_clip_model"):
            clip = ClipEncoder(cfg["clip_arch"], cfg["clip_weights"], cfg["device"], cfg["emb_batch"], cfg["num_workers"])

        with mon.stage("embed_crops") as st:
            Ef, Eu, Em = embed_all_crops(clip, rows, dirs, cfg)
            st.note(crops=3 * len(rows))

        print(f"READY: {len(rows)} detections, vectors {Ef.shape}. Cache: {dirs['out']}. Now run run_search.py.")


if __name__ == "__main__":
    main()
