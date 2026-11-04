"""Stage 2d - cut the CONFIRMED clips (strict >= match_t) and a SEPARATE REVIEW folder, each with an index and a zip."""
import csv
import os
import shutil
from collections import defaultdict

import cv2
from tqdm.auto import tqdm

from .config import mmss


def cut_all(cache, cfg, q, resq, TOP, QRES, final):
    """Returns (confirmed, review): dicts with ranges, index (list of clip rows) and zip path (None if no clips)."""
    rows, fps, stride = cache.rows, cache.fps, cache.stride
    match_t, review_min, pad_s, merge_gap = cfg["match_t"], cfg["review_min"], cfg["pad_s"], cfg["merge_gap_s"]
    trim, conf_pad, draw_boxes, draw_all = cfg["trim_to_confirmed"], cfg["conf_pad_s"], cfg["draw_boxes"], cfg["draw_all_tracks"]
    want = ("MATCH", "POSSIBLE") if cfg["include_possible"] else ("MATCH",)
    topd = {tr["key"]: tr for tr in TOP}
    verd = {x["track"]: x["verdict"] for x in final}
    dur_s = cache.nfr / fps
    tag = lambda t: f"{int(t // 60):02d}m{int(t % 60):02d}s"
    width = cfg["clip_max_width"]
    scale = min(1.0, width / cache.vw) if width else 1.0
    ow, oh = int(cache.vw * scale) // 2 * 2, int(cache.vh * scale) // 2 * 2

    def win(key, field, thr):
        """time window of a track: around the crops whose score `field` is >= thr (else the whole track)"""
        tr_ = topd[key]
        if not trim:
            return tr_["t0"], tr_["t1"]
        ts_ = [rows[d["k"]]["t"] for d in QRES[key]["frames"] if d.get(field, 0) >= thr]
        if not ts_:
            return tr_["t0"], tr_["t1"]
        return max(tr_["t0"], min(ts_) - conf_pad), min(tr_["t1"], max(ts_) + conf_pad)

    def reason(key):
        v = verd.get(key, "")
        return "unclear" if "unclear" in v else ("below threshold" if "below" in v else "confirmed")

    def cut_set(pick, field, thr, score_of, outdir, prefix, zip_name, exclude=(), max_clips=None, single=False, max_s=None):
        """merge the picked tracks into clips, cut one mp4 per clip (box on the best track), write the index and zip it"""
        items = []
        for x in final:
            if not pick(x["verdict"]):
                continue
            a_, b_ = win(x["track"], field, thr)
            if single and max_s and b_ - a_ > max_s:               # long track: keep max_s seconds centred on its best crop
                tc = rows[max(QRES[x["track"]]["frames"], key=lambda d: d["q"])["k"]]["t"]
                a_, b_ = max(a_, tc - max_s / 2), min(b_, tc + max_s / 2)
            if any(a_ <= eb and b_ >= ea for ea, eb in exclude):    # overlaps a confirmed clip: same person, skip
                continue
            items.append((a_, b_, x["track"], score_of(x)))
        items.sort(key=lambda z: (z[0], z[1]))
        clips = []
        if single:
            # one clip per track, best score first; skip a track whose window overlaps an already chosen clip by more than half of the shorter window
            for t0_, t1_, key, st in sorted(items, key=lambda z: -z[3]):
                if any(min(t1_, c["end"]) - max(t0_, c["start"]) > 0.5 * min(t1_ - t0_, c["end"] - c["start"]) for c in clips):
                    continue
                clips.append(dict(start=t0_, end=t1_, tracks=[key], best=st, best_key=key, bwin=(t0_, t1_)))
                if max_clips and len(clips) >= max_clips:
                    break
            clips.sort(key=lambda c: c["start"])
        else:
            for t0_, t1_, key, st in items:
                if clips and t0_ - clips[-1]["end"] < merge_gap:
                    c = clips[-1]
                    c["end"] = max(c["end"], t1_)
                    c["tracks"].append(key)
                    if st > c["best"]:
                        c["best_key"], c["bwin"] = key, (t0_, t1_)
                    c["best"] = max(c["best"], st)
                else:
                    clips.append(dict(start=t0_, end=t1_, tracks=[key], best=st, best_key=key, bwin=(t0_, t1_)))
            if max_clips and len(clips) > max_clips:
                clips = sorted(sorted(clips, key=lambda c: -c["best"])[:max_clips], key=lambda c: c["start"])
        print(f"{len(items)} track(s) -> {len(clips)} clip(s) after merging tracks that overlap or are < {merge_gap:g} s apart")
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir, exist_ok=True)
        out = dict(ranges=[(c["start"] - pad_s, c["end"] + pad_s) for c in clips], index=[], zip=None)
        if not clips:
            return out
        keys = {k for c in clips for k in c["tracks"]}
        dets = defaultdict(list)
        for r in rows:
            if r["track"] in keys and r["grp"] == q["target"]:
                dets[r["track"]].append(r)
        for k in dets:
            dets[k].sort(key=lambda r: r["frame"])
        cap = cv2.VideoCapture(cache.video)
        for ci, c in enumerate(tqdm(clips, desc=f"cutting {prefix} clips", unit="clip", dynamic_ncols=True)):
            a_, b_ = max(0.0, c["start"] - pad_s), min(dur_s, c["end"] + pad_s)
            f0, f1 = int(a_ * fps), int(b_ * fps)
            name = f"{prefix}_{ci + 1:03d}_{c['best']:.2f}_{tag(c['start'])}-{tag(c['end'])}.mp4"   # prefix_number_bestScore_start-end
            vw = cv2.VideoWriter(os.path.join(outdir, name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            ptr = {k: 0 for k in c["tracks"]}
            for fidx in range(f0, f1 + 1):
                ok, fr = cap.read()
                if not ok:
                    break
                if draw_boxes and ((not trim) or draw_all or (c["bwin"][0] <= fidx / fps <= c["bwin"][1])):
                    for k in (c["tracks"] if draw_all else [c["best_key"]]):
                        dl, j = dets[k], ptr[k]
                        while j + 1 < len(dl) and abs(dl[j + 1]["frame"] - fidx) <= abs(dl[j]["frame"] - fidx):
                            j += 1
                        ptr[k] = j
                        if dl and abs(dl[j]["frame"] - fidx) <= stride * 2:      # only draw while the person is actually tracked
                            x1, y1, x2, y2 = [int(v) for v in dl[j]["box"]]
                            cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(fr, f"trk {k}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(fr, ("REVIEW " if prefix == "review" else "") + mmss(fidx / fps), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if scale < 1.0:
                    fr = cv2.resize(fr, (ow, oh), interpolation=cv2.INTER_AREA)
                vw.write(fr)
            vw.release()
            out["index"].append(dict(clip=name, start=mmss(c["start"]), end=mmss(c["end"]), seconds=round(c["end"] - c["start"], 1),
                                     tracks=" ".join(str(k) for k in c["tracks"]), best_track=c["best_key"],
                                     best_score=round(c["best"], 2), reason=reason(c["best_key"])))
        cap.release()
        with open(os.path.join(outdir, "clips_index.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out["index"][0]))
            w.writeheader()
            w.writerows(out["index"])
        shutil.copy(os.path.join(resq, "final_results.json"), os.path.join(outdir, "final_results.json"))
        out["zip"] = shutil.make_archive(os.path.join(resq, zip_name), "zip", outdir)
        print("\nclip  start -> end            sec   tracks (best score, reason)")
        for r in out["index"][:cfg["print_rows"] * 2]:
            print(f"{r['clip'][:12]}  {r['start']} -> {r['end']}   {r['seconds']:>5}   {r['tracks']} ({r['best_score']}, {r['reason']})")
        print(f"ZIP: {out['zip']}  ({os.path.getsize(out['zip']) / 2 ** 20:.0f} MB, {len(clips)} clips)")
        return out

    print("=== CONFIRMED clips (strict >= %.2f) ===" % match_t)
    conf = cut_set(lambda v: v in want, "qd", match_t, lambda x: x["strict"], os.path.join(resq, "clips"), "clip", f"{q['name']}_clips")
    if not conf["index"]:
        print(f"NO CONFIRMED CLIPS: no track has strict >= {match_t:.2f} (see the REVIEW list), or check the query.")
    rev = dict(ranges=[], index=[], zip=None)
    if cfg["make_review"]:
        print("\n=== REVIEW clips (NOT confirmed, separate folder; raw >= %.2f, one clip per track, max %d clips of max %.0f s, "
              "overlaps with confirmed clips skipped) ===" % (review_min, cfg["review_max_clips"], cfg["review_max_s"]))
        rev = cut_set(lambda v: v.startswith("REVIEW"), "q", review_min, lambda x: x["raw"], os.path.join(resq, "review_clips"), "review",
                      f"{q['name']}_review", exclude=conf["ranges"], max_clips=cfg["review_max_clips"], single=True, max_s=cfg["review_max_s"])
        if not rev["index"]:
            print("NO REVIEW CLIPS.")
    return conf, rev
