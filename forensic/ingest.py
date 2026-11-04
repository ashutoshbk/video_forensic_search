"""Stage 1 - ingest: YOLO detects and tracks people and vehicles in the whole video and saves 3 crops per
detection (f = full, u = upper part, m = middle zone).

RESUMABLE: progress is saved every `checkpoint_every` sampled frames. If the run stops, start it again and it
continues where it stopped. If a tuning setting (marked * in config.yaml) changed, it starts fresh.
"""
import json
import os
import shutil
import time
from collections import defaultdict

import cv2
import torch
from tqdm.auto import tqdm

from .config import mmss

# COCO class id -> group used by the search stage
GROUP = {0: "person", 2: "vehicle", 3: "vehicle", 5: "vehicle", 7: "vehicle"}


def folder_size_gib(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 2 ** 30


def probe_video(video, max_seconds=None):
    """Read the video properties (quietly)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    used_s = min(nfr / fps, max_seconds) if max_seconds else nfr / fps
    return dict(fps=fps, frames=nfr, width=vw, height=vh, used_s=used_s)


def video_info(cfg, dirs):
    """Read the video properties and print a short summary of what is about to happen."""
    info = probe_video(cfg["video"], cfg["max_seconds"])
    fps, nfr, vw, vh, used_s = info["fps"], info["frames"], info["width"], info["height"], info["used_s"]
    print(f"video {vw}x{vh}, {fps:.1f} fps, {nfr} frames, {nfr / fps / 60:.1f} min  ->  using {used_s / 60:.1f} min")
    print(f"sampled frames to process: about {int(used_s * fps // cfg['stride'])}")
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print("GPUs:", gpus or "NONE")
    print(f"disk free in {dirs['out']}: {shutil.disk_usage(dirs['out']).free / 2 ** 30:.1f} GiB")
    print("cache folder:", dirs["out"])
    if vw > 1920:
        print("NOTE: the video is wider than 1920 px; reading and detection will be slower. stride: 10 would halve the time.")
    return info


def _read_rows_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                break                      # a half-written last line after a crash: ignore it
    return out


def run_ingest(cfg, dirs, info):
    """Detect, track and crop. Returns `rows`: one dict per detection
    (id, frame, t, track, conf, box, cid, cname, grp). Also writes meta.json (all rows) into the run folder."""
    from ultralytics import YOLO

    out, crops = dirs["out"], dirs["crops"]
    fps, nfr, vw, vh = info["fps"], info["frames"], info["width"], info["height"]
    stride, max_seconds = cfg["stride"], cfg["max_seconds"]
    classes, min_person_h, min_veh_side = cfg["ingest_classes"], cfg["min_person_h"], cfg["min_veh_side"]
    crop_max, jpeg_q = cfg["crop_max_side"], cfg["jpeg_quality"]

    rows_jsonl = os.path.join(out, "rows.jsonl")
    state_path = os.path.join(out, "state.json")
    meta_path = os.path.join(out, "meta.json")
    # fingerprint of every setting that changes the ingest result
    fp = dict(stride=stride, conf=cfg["conf"], min_person_h=min_person_h, min_veh_side=min_veh_side,
              crop_max=crop_max, jpeg_q=jpeg_q, classes=classes, w=vw, h=vh, max_seconds=max_seconds,
              yolo=cfg["yolo_weights"], imgsz=cfg["yolo_imgsz"])

    def pad_box(x1, y1, x2, y2, px, py):
        return (max(0, int(x1 - px)), max(0, int(y1 - py)), min(vw, int(x2 + px)), min(vh, int(y2 + py)))

    def save_crop(img, a, path):
        x1, y1, x2, y2 = a
        x2, y2 = max(x2, x1 + 2), max(y2, y1 + 2)
        crop = img[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        m = max(h, w)
        if m > crop_max:
            sc = crop_max / m
            crop = cv2.resize(crop, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])

    # ---- decide: finished / resume / fresh start ----
    state = None
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    if state is not None and state.get("fp") != fp:
        print("The settings changed since the saved ingest (stride, thresholds, classes or time limit). Starting fresh.")
        for p in (state_path, rows_jsonl, meta_path):
            if os.path.exists(p):
                os.remove(p)
        state = None

    rows, id_offset, start_frame, done = [], 0, 0, False
    if state is not None:
        start_frame, done = int(state["next_frame"]), bool(state.get("done"))
        rows = [r for r in _read_rows_jsonl(rows_jsonl) if r["frame"] < start_frame]   # drop rows of an unfinished frame
        if not done:
            with open(rows_jsonl, "w") as f:                                          # rewrite the consistent part
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            tids = [r["track"] for r in rows if r["track"] >= 0]
            id_offset = (max(tids) + 1000) if tids else 0     # new track ids after a restart cannot clash with old ones
    elif os.path.exists(rows_jsonl):
        os.remove(rows_jsonl)

    last_frame = min(nfr - 1, int(max_seconds * fps)) if max_seconds else nfr - 1

    if done:
        print(f"ingest already finished: {len(rows)} detections loaded from the cache")
    else:
        if start_frame:
            print(f"RESUMING at {mmss(start_frame / fps)} ({len(rows)} detections already saved)")
        t0 = time.time()
        yolo = YOLO(cfg["yolo_weights"])
        cap = cv2.VideoCapture(cfg["video"])
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_steps = max(1, (last_frame - start_frame) // stride + 1)
        pbar = tqdm(total=total_steps, desc="1/2 detect + crop", unit="frame", dynamic_ncols=True, smoothing=0.1)
        pending, fidx, steps = [], start_frame, 0

        def flush(next_frame, finished=False):
            with open(rows_jsonl, "a") as f:
                for r in pending:
                    f.write(json.dumps(r) + "\n")
            pending.clear()
            with open(state_path, "w") as f:
                json.dump(dict(fp=fp, next_frame=int(next_frame), done=finished), f)

        try:
            while fidx <= last_frame:
                ok, img = cap.read()
                if not ok:
                    break
                r = yolo.track(img, persist=True, tracker="bytetrack.yaml", classes=classes, conf=cfg["conf"],
                               imgsz=cfg["yolo_imgsz"], device=cfg["device"], verbose=False)[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
                    cids = r.boxes.cls.int().cpu().tolist()
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    for b, tid, cid, cf in zip(xyxy, ids, cids, confs):
                        x1, y1, x2, y2 = [float(v) for v in b]
                        h, w = y2 - y1, x2 - x1
                        grp = GROUP.get(cid, "vehicle")
                        if grp == "person" and h < min_person_h:
                            continue
                        if grp == "vehicle" and min(w, h) < min_veh_side:
                            continue
                        n = len(rows)
                        p = os.path.join(crops, f"{n:06d}")
                        if grp == "person":
                            # f = full person, u = head + upper body (top 55%), m = chest / hands zone (25%-75%)
                            save_crop(img, pad_box(x1, y1, x2, y2, 0.08 * w, 0.05 * h), p + "_f.jpg")
                            save_crop(img, pad_box(x1, y1, x2, y1 + 0.55 * h, 0.15 * w, 0.05 * h), p + "_u.jpg")
                            save_crop(img, pad_box(x1, y1 + 0.25 * h, x2, y1 + 0.75 * h, 0.20 * w, 0.0), p + "_m.jpg")
                        else:
                            # f = whole vehicle, u = upper 60%, m = centre zoom (70% x 70%)
                            save_crop(img, pad_box(x1, y1, x2, y2, 0.06 * w, 0.06 * h), p + "_f.jpg")
                            save_crop(img, pad_box(x1, y1, x2, y1 + 0.60 * h, 0.05 * w, 0.05 * h), p + "_u.jpg")
                            save_crop(img, pad_box(x1 + 0.15 * w, y1 + 0.15 * h, x2 - 0.15 * w, y2 - 0.15 * h, 0, 0), p + "_m.jpg")
                        row = dict(id=n, frame=fidx, t=fidx / fps,
                                   track=(int(tid) + id_offset) if tid >= 0 else -1,
                                   conf=float(cf), box=[x1, y1, x2, y2], cid=int(cid),
                                   cname=r.names[int(cid)], grp=grp)
                        rows.append(row)
                        pending.append(row)
                for _ in range(stride - 1):
                    cap.grab()                                # skip frames cheaply
                fidx += stride
                steps += 1
                pbar.update(1)
                if steps % 20 == 0:
                    npers = sum(1 for q in rows[-2000:] if q["grp"] == "person")
                    pbar.set_postfix(detections=len(rows), video_time=mmss(fidx / fps), recent_people=npers)
                if steps % cfg["checkpoint_every"] == 0:
                    flush(fidx)
            flush(fidx, finished=True)
        except BaseException:
            flush(fidx)                                       # keep what is done; run again to continue
            pbar.close()
            print(f"\nStopped at {mmss(fidx / fps)}. Progress saved ({len(rows)} detections). Run the same command again to continue.")
            raise
        pbar.close()
        cap.release()
        with open(meta_path, "w") as f:
            json.dump(rows, f)
        print(f"ingest done: {len(rows)} detections, {len(set(r['track'] for r in rows))} tracks, "
              f"{mmss(time.time() - t0)} (h:m:s or m:s)")
        del yolo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(rows) == 0:
        raise RuntimeError("Nothing detected. Lower conf / min_person_h or check the video path.")
    cnt = defaultdict(int)
    for r in rows:
        cnt[r["cname"]] += 1
    print("detections by class:", dict(cnt))
    print(f"crops folder size: {folder_size_gib(crops):.2f} GiB")
    return rows


def load_rows(dirs):
    """Load the detections of a FINISHED ingest (used by the search stage)."""
    state_path = os.path.join(dirs["out"], "state.json")
    if not os.path.exists(state_path):
        raise RuntimeError(f"no ingest found in {dirs['out']}: run `python run_ingest.py` first")
    with open(state_path) as f:
        if not json.load(f).get("done"):
            raise RuntimeError("the ingest is not finished: run `python run_ingest.py` again to continue it")
    meta = os.path.join(dirs["out"], "meta.json")
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f)
    return _read_rows_jsonl(os.path.join(dirs["out"], "rows.jsonl"))
