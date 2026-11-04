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
import os

from forensic.clips import cut_all
from forensic.config import load_config, load_query, mmss, run_dirs
from forensic.decide import decide
from forensic.embed import ClipEncoder
from forensic.monitor import RunMonitor
from forensic.reference import load_reference
from forensic.search import IngestCache, shortlist
from forensic.verify import Qwen, check_tracks, plan_and_save


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", help="overrides `video` in the config")
    ap.add_argument("--max-seconds", type=float, help="overrides `max_seconds` in the config")
    ap.add_argument("--query", nargs="+", help="one or more query files (see query.yaml)")
    ap.add_argument("--text", help="a query written on the command line instead of a file")
    ap.add_argument("--name", help="results folder name for --text")
    ap.add_argument("--target", choices=["person", "vehicle"])
    ap.add_argument("--start", type=float, help="search from this video second")
    ap.add_argument("--end", type=float, help="search until this video second")
    ap.add_argument("--reference", help="path of a reference photo")
    args = ap.parse_args()

    cfg = load_config(args.config, {"video": args.video, "max_seconds": args.max_seconds})
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
        with mon.stage("load_clip_model"):
            clip = ClipEncoder(cfg["clip_arch"], cfg["clip_weights"], cfg["device"], cfg["emb_batch"], cfg["num_workers"])
        qwen = None

        for q in queries:
            name = q["name"]
            print(f"\n{'=' * 70}\nQUERY [{name}]: {q['text']}  (target: {q['target']})\n{'=' * 70}")
            resq = os.path.join(dirs["results"], name)
            os.makedirs(resq, exist_ok=True)

            with mon.stage("reference_photo", name):
                ref_pil, ref_vec = load_reference(q["reference"], clip, q["require_reference"])

            if qwen is None:
                with mon.stage("load_qwen"):
                    qwen = Qwen(cfg)

            with mon.stage("plan_query", name):
                attr, same_q = plan_and_save(qwen, q, resq)

            with mon.stage("shortlist", name) as st:
                sl = shortlist(cache, cfg, q, clip, ref_pil, ref_vec, resq)
                st.note(tracks_shortlisted=len(sl["TOP"]))

            with mon.stage("qwen_checks", name) as st:
                QRES = check_tracks(qwen, cache, cfg, sl["TOP"], attr, same_q, ref_pil)
                st.note(tracks=len(sl["TOP"]), crops=sum(len(t["crops"]) for t in sl["TOP"]))

            with mon.stage("decision", name) as st:
                final = decide(cache, cfg, q, resq, sl["TOP"], QRES)
                st.note(match=sum(x["verdict"] == "MATCH" for x in final),
                        review=sum(x["verdict"].startswith("REVIEW") for x in final))

            with mon.stage("clips", name):
                cut_all(cache, cfg, q, resq, sl["TOP"], QRES, final)

            print(f"[time] QUERY TOTAL [{name}]: {mmss(mon.group_seconds(name))}. Results in: {resq}")


if __name__ == "__main__":
    main()
