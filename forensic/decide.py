"""Stage 2c - final decision per track: MATCH / POSSIBLE / REVIEW / insufficient evidence."""
import json
import os
import time

import cv2
import numpy as np

from .config import mmss
from .search import plt, save_hit_frames, size_px, too_small


def decide(cache, cfg, q, resq, TOP, QRES):
    """Returns `final`: one dict per checked track, judged tracks first (best rank score first)."""
    rows = cache.rows
    qw, match_t, possible_t, review_min = cfg["qw"], cfg["match_t"], cfg["possible_t"], cfg["review_min"]
    clip_scaled = {tr["key"]: 1 / (1 + np.exp(-tr["score"] / 2)) for tr in TOP}
    final = []
    for tr in TOP:
        d = QRES[tr["key"]]
        k = d["best_k"]
        f = qw * d["q_raw"] + (1 - qw) * clip_scaled[tr["key"]]       # ranking uses the raw score so vetoed tracks still sort by how close they were
        if too_small(rows, k, cfg):
            verdict = "insufficient evidence (too small)"
        elif d["q"] >= match_t:
            verdict = "MATCH"
        elif d["q"] >= possible_t:
            verdict = "POSSIBLE"
        elif d["q_raw"] >= review_min:
            verdict = "REVIEW (unclear)" if d["any_unclear"] else f"REVIEW (below {match_t:.2f})"   # not confirmed: goes to the separate review list
        else:
            verdict = "insufficient evidence"
        final.append(dict(track=tr["key"], cls=rows[k]["cname"], first=mmss(tr["t0"]), last=mmss(tr["t1"]),
                          best_t=mmss(rows[k]["t"]), px=int(size_px(rows, k)), strict=round(d["q"], 3), raw=round(d["q_raw"], 3),
                          avg=round(d["q_avg"], 3), clip=round(float(clip_scaled[tr["key"]]), 3), final=round(float(f), 3),
                          verdict=verdict, best_k=k))
    final.sort(key=lambda x: (x["verdict"].endswith("(too small)"), -x["final"]))   # judged tracks first

    print("rank  track  class       seen          best_t  px    STRICT  raw    avg    clip   rank-score  verdict")
    for i, x in enumerate(final[:cfg["print_rows"]], 1):
        print(f"{i:>4}  {str(x['track']):>5}  {x['cls']:<10}  {x['first']}-{x['last']}   {x['best_t']}   {x['px']:>4}  {x['strict']:.2f}    "
              f"{x['raw']:.2f}   {x['avg']:.2f}   {x['clip']:.2f}   {x['final']:.2f}        {x['verdict']}")

    judged = [x for x in final if not x["verdict"].endswith("(too small)")]
    if judged:
        print(f"\nCALIBRATION HINT: highest STRICT score among judged tracks = {max(x['strict'] for x in judged):.2f}. "
              "Positive query: match_t must stay below the true target's STRICT score; negative query: above the best wrong score.")

    with open(os.path.join(resq, "final_results.json"), "w") as f:
        json.dump([{k: v for k, v in x.items() if k != "best_k"} for x in final], f, indent=1)
    verd = {x["track"]: x["verdict"] for x in final}
    last = TOP[int(0.75 * len(TOP)):]
    m_last = sum(1 for tr in last if verd.get(tr["key"]) == "MATCH")
    print(f"\nVERDICTS over all {len(final)} checked tracks: MATCH={sum(v == 'MATCH' for v in verd.values())}, "
          f"POSSIBLE={sum(v == 'POSSIBLE' for v in verd.values())}, REVIEW={sum(v.startswith('REVIEW') for v in verd.values())}, "
          f"rest={sum(not (v in ('MATCH', 'POSSIBLE') or v.startswith('REVIEW')) for v in verd.values())}")
    print(f"SATURATION CHECK: {m_last} MATCH track(s) among the last {len(last)} shortlisted (lowest OpenCLIP rank). "
          + ("Near 0 -> the shortlist is wide enough." if m_last <= max(1, 0.02 * len(last))
             else "NOT near 0 -> matches may exist outside the shortlist: raise topk and search again."))

    # boxed frames of the final top 6 (best frame per track, read from the original video)
    _t = time.perf_counter()
    top_final = [dict(best=x["best_k"]) for x in final]
    if cfg.get("save_debug_images", False): save_hit_frames(cache, resq, top_final, "final", scores_txt=[f"{x['final']:.2f} {x['verdict'][:5]}" for x in final])
    print(f"[dbg] save_hit_frames {time.perf_counter()-_t:.2f}s", flush=True); _t = time.perf_counter()

    # the crops Qwen actually judged for the top 6 tracks, so you can see what "best crops" picked
    n_crops = cfg["n_crops"]
    show = final[:6]
    if show and cfg.get("save_debug_images", False):
        fig, axes = plt.subplots(len(show), n_crops, figsize=(2.2 * n_crops, 3.4 * len(show)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, x in enumerate(show):
            for j, d in enumerate(sorted(QRES[x["track"]]["frames"], key=lambda d: -d["q"])[:n_crops]):
                k = d["k"]
                im = cv2.imread(os.path.join(cache.crops, f"{rows[k]['id']:06d}_f.jpg"))
                if im is not None:
                    axes[i][j].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                axes[i][j].set_title(f"trk {x['track']} {mmss(rows[k]['t'])}\nqwen {d['q']:.2f}  {int(size_px(rows, k))}px", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(resq, "best_crops_top6.png"), dpi=100)
        plt.close(fig)
    print(f"[dbg] best_crops_png {time.perf_counter()-_t:.2f}s", flush=True)

    n_match = sum(1 for x in final if x["verdict"] == "MATCH")
    n_poss = sum(1 for x in final if x["verdict"] == "POSSIBLE")
    top_strict = max([x["strict"] for x in judged], default=0.0)
    print(f"\nSEARCH SUMMARY [{q['name']}]: MATCH={n_match}, POSSIBLE={n_poss}, best strict score among judged tracks={top_strict:.2f}")
    rev = sorted([x for x in final if x["verdict"].startswith("REVIEW")], key=lambda x: -x["raw"])
    if rev:
        print(f"REVIEW LIST ({len(rev)} tracks, not confirmed; the best {cfg['review_max_clips']} are cut into a separate review folder): " +
              ", ".join(f"trk {x['track']} {x['first']}-{x['last']} (raw {x['raw']:.2f})" for x in rev[:15]))
    if n_match + n_poss == 0:
        print("NOTHING above the match line. Negative query: this is the expected answer. Positive query: open shortlist_crops.png, "
              "check whether the target is in the shortlist at all (if yes, read its strict score in the table above; if no, raise topk or "
              "widen the time window).")
    else:
        print("FOUND something: open final_frames.png and check by eye whether the top hit is the target. Note its strict score.")
    return final
