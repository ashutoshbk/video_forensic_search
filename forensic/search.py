"""Stage 2a - shortlist: score every detection with OpenCLIP (reference photo + text), filter, group by track,
keep the best TOPK tracks and choose the best crops of each track for the Qwen checks."""
import os
from collections import defaultdict

import cv2
import matplotlib
import numpy as np
from tqdm.auto import tqdm

from .config import mmss
from .embed import load_embeddings
from .ingest import load_rows, probe_video

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class IngestCache:
    """Everything the search needs from a finished ingest: detections, vectors and video properties."""

    def __init__(self, cfg, dirs):
        self.cfg, self.dirs, self.video, self.crops = cfg, dirs, cfg["video"], dirs["crops"]
        info = probe_video(cfg["video"], cfg["max_seconds"])
        self.fps, self.nfr, self.vw, self.vh = info["fps"], info["frames"], info["width"], info["height"]
        self.stride = cfg["stride"]
        self.rows = load_rows(dirs)
        n_fix = 0
        for r in self.rows:             # older caches may lack the group: rebuild it from the class
            if "grp" not in r:
                r["grp"] = "person" if (r.get("cname") == "person" or r.get("cid") == 0) else "vehicle"
                n_fix += 1
        if n_fix:
            print(f"NOTE: {n_fix} rows had no 'grp' - rebuilt it from the class name")
        self.Ef, self.Eu, self.Em = load_embeddings(dirs, cfg, len(self.rows))
        print(f"loaded ingest: {len(self.rows)} detections, vectors {self.Ef.shape}, {info['used_s'] / 60:.1f} min of video")


def size_px(rows, k):
    x1, y1, x2, y2 = rows[k]["box"]
    return (y2 - y1) if rows[k]["grp"] == "person" else min(x2 - x1, y2 - y1)


def too_small(rows, k, cfg):
    return size_px(rows, k) < (cfg["min_judge_person_h"] if rows[k]["grp"] == "person" else cfg["min_judge_veh_side"])


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def _robust_z(x, mask):
    v = x[mask]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return (x - med) / mad


def save_hit_frames(cache, resq, top, tag, scores_txt=None):
    """Boxed original frames of the first 6 tracks of `top` (read from the video) -> <tag>_frames.png."""
    rows = cache.rows
    cap = cv2.VideoCapture(cache.video)
    paths = []
    for j, tr in enumerate(top[:6], 1):
        r = rows[tr["best"]]
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, fr = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 0, 255), 4)
        lab = f"#{j} {r['cname']} trk{r['track']} {mmss(r['t'])}" + (f" {scores_txt[j - 1]}" if scores_txt else "")
        cv2.putText(fr, lab, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        p = os.path.join(resq, f"{tag}_hit{j}_trk{r['track']}_{int(r['t'])}s.jpg")
        cv2.imwrite(p, fr)
        paths.append(p)
    cap.release()
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p in zip(axes.ravel(), paths):
        ax.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.savefig(os.path.join(resq, f"{tag}_frames.png"), dpi=90)
    plt.close(fig)


def shortlist(cache, cfg, q, clip, ref_pil, ref_vec, resq):
    """Returns dict(TOP=..., groups=...). TOP = shortlisted tracks, each with the crops (row ids) Qwen will check."""
    rows, target = cache.rows, q["target"]
    w_img, w_txt = cfg["w_img"], cfg["w_txt"]
    if ref_vec is None:
        w_img, w_txt = 0.0, 1.0
    filters = dict(edge=True, static=q["filter_static"], fixed=q["filter_fixed"]) if target == "person" \
        else dict(edge=False, static=False, fixed=False)       # vehicles are often parked or enter from the edge

    t_lo = q["search_start"] if q["search_start"] is not None else 0
    t_hi = q["search_end"] if q["search_end"] is not None else 1e12
    sel = np.array([(r["grp"] == target) and (t_lo <= r["t"] <= t_hi) for r in rows])
    print(f"{sel.sum()} of {len(rows)} detections are '{target}' between {mmss(t_lo)} and {mmss(min(t_hi, cache.nfr / cache.fps))}")
    if sel.sum() == 0:
        raise RuntimeError("No detections of this target type in this time window.")

    pos_v = clip.text_vec([q["text"]])
    views = [cache.Ef, cache.Eu, cache.Em]
    txt_raw = np.max([E @ pos_v for E in views], axis=0)          # no negative prompts: positive text only
    img_raw = np.max([E @ ref_vec for E in views], axis=0) if ref_vec is not None else np.zeros(len(rows))

    # --- valid-detection filters ---
    valid = sel.copy()
    by_track = defaultdict(list)
    for k, r in enumerate(rows):
        if sel[k]:
            by_track[r["track"]].append(k)

    if filters["static"]:
        static_tracks = set()
        for tid, ks in by_track.items():
            if tid < 0 or len(ks) < cfg["min_det_static"]:
                continue
            cx = np.array([(rows[k]["box"][0] + rows[k]["box"][2]) / 2 for k in ks])
            cy = np.array([(rows[k]["box"][1] + rows[k]["box"][3]) / 2 for k in ks])
            hh = np.median([rows[k]["box"][3] - rows[k]["box"][1] for k in ks])
            if max(np.ptp(cx), np.ptp(cy)) < cfg["static_rel"] * hh:
                static_tracks.add(tid)
                print(f"static track {tid}: ({cx.mean():.0f},{cy.mean():.0f}) over {len(ks)} detections -> dropped")
        for k, r in enumerate(rows):
            if r["track"] in static_tracks:
                valid[k] = False

    if filters["edge"]:
        for k, r in enumerate(rows):
            x1, y1, x2, y2 = r["box"]
            if (x1 <= cfg["edge_px"] or x2 >= cache.vw - cfg["edge_px"]) and (x2 - x1) < cfg["edge_aspect"] * (y2 - y1):
                valid[k] = False

    if filters["fixed"]:
        idx = np.where(valid)[0]
        B = np.array([rows[k]["box"] for k in idx], dtype=np.float32)
        T = np.array([rows[k]["t"] for k in idx], dtype=np.float32)
        area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        drop = 0
        for a in range(len(idx)):
            ix1 = np.maximum(B[a, 0], B[:, 0]); iy1 = np.maximum(B[a, 1], B[:, 1])
            ix2 = np.minimum(B[a, 2], B[:, 2]); iy2 = np.minimum(B[a, 3], B[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            iou_a = inter / (area[a] + area - inter + 1e-6)
            if np.sum((iou_a > cfg["fixed_iou"]) & (np.abs(T - T[a]) > cfg["fixed_dt"])) >= cfg["fixed_min"]:
                valid[idx[a]] = False
                drop += 1
        print(f"fixed-position boxes dropped: {drop}")
    print(f"kept {valid.sum()} of {sel.sum()} {target} detections")
    if valid.sum() == 0:
        raise RuntimeError("All detections were filtered out. Relax filter_static / filter_fixed.")

    z_img = _robust_z(img_raw, valid) if w_img > 0 else np.zeros(len(rows))
    z_txt = _robust_z(txt_raw, valid)
    score = w_img * z_img + w_txt * z_txt
    score[~valid] = -99

    # --- group by track, score each track on its best n_agg detections (big enough ones first) ---
    groups = defaultdict(list)
    for k, r in enumerate(rows):
        if valid[k]:
            groups[r["track"] if r["track"] >= 0 else f"solo{k}"].append(k)
    tracks = []
    for key, ks in groups.items():
        if len(ks) < cfg["min_track_det"]:
            continue
        ks_sorted = sorted(ks, key=lambda k: -score[k])
        ks_big = [k for k in ks_sorted if not too_small(rows, k, cfg)]
        small = len(ks_big) == 0                              # never big enough to judge
        use = ks_sorted if small else ks_big
        top = use[:cfg["n_agg"]]
        ts = [rows[k]["t"] for k in ks]
        tracks.append(dict(key=key, best=use[0], top=top, score=float(np.mean([score[k] for k in top])),
                           t0=min(ts), t1=max(ts), n=len(ks), small=small,
                           txt_max=float(max(txt_raw[k] for k in ks))))
    tracks.sort(key=lambda d: (d["small"], -d["score"]))      # judged tracks first, small ones after them
    if not tracks:
        raise RuntimeError("No track has enough detections (min_track_det).")

    TOP = []
    for tr in tracks:                                         # same object split into two tracks: keep one
        rb = rows[tr["best"]]
        if any(abs(rb["t"] - rows[q_["best"]]["t"]) <= cfg["dedup_sec"] and _iou(rb["box"], rows[q_["best"]]["box"]) > cfg["dedup_iou"]
               for q_ in TOP):
            continue
        TOP.append(tr)
    # hybrid selection: every track with score >= z_min, at least min_tracks, at most topk
    z_min = cfg.get("z_min")
    n_all = len(TOP)
    if z_min is None:                                         # old behaviour if the yaml has no z_min
        TOP = TOP[:cfg["topk"]]
    else:
        min_tracks = cfg.get("min_tracks", 25)
        floor_v = cfg.get("abs_floor_" + target, cfg.get("abs_floor"))
        if floor_v is None:
            floor_v = cfg.get("abs_floor")
        ok = lambda tr: floor_v is None or tr["txt_max"] >= floor_v     # floor applies to both gates
        keep = [i for i, tr in enumerate(TOP) if tr["score"] >= z_min and ok(tr)]
        if len(keep) < min_tracks:                            # pad with the next best, in ranked order
            ks = set(keep)
            z_pad = cfg.get("z_pad", 0.5)                        # padding only from tracks scoring at least this
            keep += [i for i in range(n_all) if i not in ks and TOP[i]["score"] >= z_pad and ok(TOP[i])][:min_tracks - len(keep)]
            keep.sort()
        TOP = [TOP[i] for i in keep[:cfg["topk"]]]
        print(f"[threshold] z_min={z_min}: sent {len(TOP)} of {n_all} de-duplicated tracks (min_tracks={min_tracks}, topk={cfg['topk']})")

    # all filtered tracks -> csv for calibration (sent = went to Qwen)
    import csv
    sent_keys = {str(t["key"]) for t in TOP}
    with open(os.path.join(resq, "shortlist_scores.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["track", "rank", "score", "txt_max", "sent"])
        for i, t in enumerate(tracks, 1):
            w.writerow([t["key"], i, round(t["score"], 3), round(t["txt_max"], 4), str(t["key"]) in sent_keys])

    # absolute floor on raw OpenCLIP similarity (None = off). Empty result is allowed: the video is skipped.
    floor = cfg.get("abs_floor_" + target, cfg.get("abs_floor"))
    if floor is None:
        floor = cfg.get("abs_floor")
    if floor is not None:
        kept = [t for t in TOP if t["txt_max"] >= floor]
        print(f"[floor] abs_floor={floor}: {len(kept)} of {len(TOP)} tracks pass")
        TOP = kept

    # ---- best crops: chosen by SIZE + SHARPNESS, not only by the OpenCLIP score ----
    n_crops = cfg["n_crops"]

    def sharp_of(k):
        """Laplacian variance of the saved full-body crop at a fixed size (long side 128 px), so it is comparable."""
        im = cv2.imread(os.path.join(cache.crops, f"{rows[k]['id']:06d}_f.jpg"), cv2.IMREAD_GRAYSCALE)
        if im is None or im.size == 0:
            return 0.0
        h, w = im.shape
        sc = 128.0 / max(h, w)
        im = cv2.resize(im, (max(8, int(w * sc)), max(8, int(h * sc))), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        return float(cv2.Laplacian(im, cv2.CV_64F).var())

    for tr in tqdm(TOP, desc="best crops", unit="track", dynamic_ncols=True):
        if cfg["crop_mode"] != "quality":
            tr["crops"] = list(tr["top"][:3])
            continue
        ks = groups[tr["key"]]
        big = [k for k in ks if not too_small(rows, k, cfg)]
        cands = sorted(big if big else ks, key=lambda k: -size_px(rows, k))[:cfg["cand_per_track"]]
        sh = {k: sharp_of(k) for k in cands}
        smax, zmax = max(sh.values()) + 1e-6, max(size_px(rows, k) for k in cands) + 1e-6
        qual = {k: cfg["w_size"] * size_px(rows, k) / zmax + cfg["w_sharp"] * sh[k] / smax for k in cands}
        order = sorted(cands, key=lambda k: -qual[k])
        chosen = []
        for k in order:
            if all(abs(rows[k]["t"] - rows[c]["t"]) >= cfg["min_crop_gap_s"] for c in chosen):
                chosen.append(k)
            if len(chosen) == n_crops:
                break
        for k in order:                                       # fewer than n_crops far enough apart: fill with the next best
            if len(chosen) == n_crops:
                break
            if k not in chosen:
                chosen.append(k)
        tr["crops"] = chosen
    print(f"[best crops] mode={cfg['crop_mode']}: {sum(len(t['crops']) for t in TOP)} crops for {len(TOP)} tracks")

    print("\nrank  track  class       seen (first-last)  n_det  best_t  size   score   z_img  z_txt  img_cos  judged?")
    print(f"tracks passing the filters in this window: {len(tracks)}; shortlisted: {len(TOP)} (lowest shortlisted score {(TOP[-1]['score'] if TOP else float('nan')):+.2f})")
    for rank, tr in enumerate(TOP[:cfg["print_rows"]], 1):
        k = tr["best"]
        print(f"{rank:>4}  {str(tr['key']):>5}  {rows[k]['cname']:<10}  {mmss(tr['t0'])}-{mmss(tr['t1'])}        {tr['n']:>4}  {mmss(rows[k]['t'])}  "
              f"{int(size_px(rows, k)):>4}  {tr['score']:+.2f}  {z_img[k]:+.2f}  {z_txt[k]:+.2f}  {img_raw[k]:.3f}  {'too small' if tr['small'] else 'yes'}")

    # contact sheet: reference photo first, then the best crop of each ranked track
    show = TOP[:cfg["contact_n"]]
    n = len(show) + (1 if ref_pil is not None else 0)
    cols = 10
    nrows = max(1, -(-n // cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(2.4 * cols, 4.0 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    j0 = 0
    if ref_pil is not None:
        axes[0].imshow(ref_pil)
        axes[0].set_title("REFERENCE", fontsize=8)
        j0 = 1
    for j, tr in enumerate(show):
        r = rows[tr["best"]]
        im = cv2.imread(os.path.join(cache.crops, f"{r['id']:06d}_f.jpg"))
        if im is not None:
            axes[j + j0].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[j + j0].set_title(f"#{j + 1} {r['cname']} trk {tr['key']}\n{mmss(r['t'])}  {tr['score']:+.1f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(resq, "shortlist_crops.png"), dpi=110)
    plt.close(fig)
    save_hit_frames(cache, resq, TOP, "clip")
    return dict(TOP=TOP, groups=groups)
