"""Settings loading (config.yaml = pipeline settings, query.yaml = one search) and the folder layout of a run."""
import os
import re

import yaml

DEFAULTS = dict(
    video=None, output_dir="outputs", max_seconds=None, device="cuda:0",
    # ---- ingest ----
    yolo_weights="yolo11m.pt", yolo_imgsz=1280, stride=5, conf=0.30,
    ingest_classes=[0, 2, 3, 5, 7], min_person_h=100, min_veh_side=80,
    crop_max_side=512, jpeg_quality=90, checkpoint_every=300,
    clip_arch="ViT-L-14", clip_weights="laion2b_s32b_b82k",
    emb_chunk=8192, emb_batch=128, num_workers=min(4, os.cpu_count() or 2),
    # ---- search: shortlist ----
    w_img=0.6, w_txt=0.4, n_agg=3, min_track_det=2, topk=300, z_min=None, min_tracks=25, z_pad=0.5, abs_floor=None, abs_floor_person=None, abs_floor_vehicle=None, print_rows=30, contact_n=40,
    static_rel=0.03, min_det_static=4, edge_px=4, edge_aspect=0.30,
    fixed_iou=0.85, fixed_dt=2.0, fixed_min=6, dedup_sec=2.0, dedup_iou=0.30,
    min_judge_person_h=100, min_judge_veh_side=80,
    n_crops=5, crop_mode="quality", cand_per_track=40, min_crop_gap_s=0.5, w_size=0.5, w_sharp=0.5,
    # ---- search: Qwen2-VL ----
    qwen_model="Qwen/Qwen2-VL-7B-Instruct", qwen_4bit=False, qwen_max_memory=None,
    qwen_min_pixels=128 * 28 * 28, qwen_max_pixels=512 * 28 * 28, qwen_batch=16,
    # ---- search: decision ----
    qw=0.7, match_t=0.90, possible_t=0.90, review_min=0.60,
    # ---- search: clips ----
    include_possible=False, pad_s=1.5, merge_gap_s=2.0, clip_max_width=960, draw_boxes=True,
    trim_to_confirmed=True, conf_pad_s=3.0, draw_all_tracks=False,
    make_review=True, review_max_clips=15, review_max_s=10.0,
)

QUERY_DEFAULTS = dict(
    text=None,                 # the ONLY thing you must write: e.g. "a person with tattoo"
    name=None,                 # results go to <run folder>/results/<name>/ (default: made from the text)
    target="person",           # "person" or "vehicle"
    search_start=0,            # only detections between these video seconds are searched
    search_end=None,           # None = until the end of the video
    filter_static=False,       # drop boxes that never move (statues); OFF for people who stand still
    filter_fixed=False,        # drop boxes that reappear at the same place (posters); slow on long windows
    reference=None,            # path to a reference photo, or None for a text-only search
    require_reference=False,   # True = stop if the reference photo is missing
)


def _read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(defaults, user, overrides, source):
    unknown = set(user) - set(defaults)
    if unknown:
        raise ValueError(f"unknown setting(s) in {source}: {sorted(unknown)}")
    out = {**defaults, **user}
    for k, v in (overrides or {}).items():
        if v is not None:
            out[k] = v
    return out


def load_config(path="config.yaml", overrides=None):
    """Read config.yaml, fill missing keys with defaults, then apply command-line overrides (None = ignore)."""
    cfg = _merge(DEFAULTS, _read_yaml(path), overrides, path)
    if not cfg["video"]:
        raise ValueError("no video set: put `video:` in config.yaml or use --video")
    return cfg


def load_query(path=None, overrides=None):
    """Read one search (query.yaml style file, or None to start from the defaults) and apply overrides."""
    q = _merge(QUERY_DEFAULTS, _read_yaml(path) if path else {}, overrides, path or "the command line")
    if not q["text"]:
        raise ValueError(f"no query text in {path or 'the command line'}: set `text:` (or use --text)")
    if q["target"] not in ("person", "vehicle"):
        raise ValueError("target must be 'person' or 'vehicle'")
    if not q["name"]:
        q["name"] = re.sub(r"\W+", "_", q["text"].lower()).strip("_")[:40] or "query"
    return q


def run_dirs(cfg):
    """One folder per video and time limit, so different videos never mix. Created if missing."""
    stem = os.path.splitext(os.path.basename(cfg["video"]))[0]
    tag = "full" if not cfg["max_seconds"] else f"{cfg['max_seconds']:g}s"
    out = os.path.join(cfg["output_dir"], f"{stem}_{tag}")
    dirs = dict(out=out, crops=os.path.join(out, "crops"), results=os.path.join(out, "results"))
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def mmss(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"
