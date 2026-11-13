import csv, glob, json, os, sys
name = sys.argv[1]
conf, rej = [], []
for d in glob.glob(f"outputs/*/results/{name}"):
    f = os.path.join(d, "shortlist_scores.csv"); j = os.path.join(d, "clips", "final_results.json")
    if not (os.path.isfile(f) and os.path.isfile(j)): continue
    verd = {str(x["track"]): x["verdict"] for x in json.load(open(j))}
    vid = os.path.basename(os.path.dirname(os.path.dirname(d)))
    for r in csv.DictReader(open(f)):
        if r["sent"] != "True": continue
        v = verd.get(r["track"], "")
        row = (vid, r["track"], int(r["rank"]), float(r["score"]), float(r["txt_max"]), v)
        (conf if v == "MATCH" else rej).append(row)
print("CONFIRMED (MATCH): video, track, rank, z-score, txt_max")
for r in sorted(conf, key=lambda r: r[4]): print("  ", r[:5])
if conf:
    print("lowest txt_max among MATCH:", min(r[4] for r in conf), "| worst rank:", max(r[2] for r in conf))
if rej:
    t = sorted(r[4] for r in rej); n = len(t)
    print(f"rejected: {n} tracks, txt_max min {t[0]:.4f}, median {t[n//2]:.4f}, 90th pct {t[int(n*.9)]:.4f}, max {t[-1]:.4f}")
