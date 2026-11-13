import re, glob, os, sys, datetime
arg = sys.argv[1]
files = [arg] if os.path.isfile(arg) else sorted(glob.glob(f"outputs/_ui_logs/{arg}_*.log"))
def secs(s):
    m = re.match(r'(?:(\d+)m)?([\d.]+)s$', s)
    return int(m.group(1) or 0) * 60 + float(m.group(2))
def mmss(s):
    return sum(int(v) * 60 ** i for i, v in enumerate(reversed(s.split(':'))))
tot, sent, match, n_q = {}, 0, 0, 0
for f in files:
    t = open(f, errors="ignore").read()
    for k, v in re.findall(r"\[time\] (load_qwen|load_clip_model|load_ingest_cache): ([\dm.]+s)", t):
        tot[k] = tot.get(k, 0) + secs(v)
    for v in re.findall(r"qwen_checks: ([\dm.]+s)", t):
        tot["qwen_checks"] = tot.get("qwen_checks", 0) + secs(v)
    for v in re.findall(r"QUERY TOTAL \[[^\]]*\]: ([\d:]+)", t):
        tot["query_total"] = tot.get("query_total", 0) + mmss(v); n_q += 1
    sent += sum(int(x) for x in re.findall(r"shortlisted: (\d+)", t))
    match += sum(int(x) for x in re.findall(r"SEARCH SUMMARY \[[^\]]*\]: MATCH=(\d+)", t))
mt = [os.path.getmtime(f) for f in files]
print(f"{arg}: {len(files)} log file(s), {n_q} video queries")
print(f"  load_qwen {tot.get('load_qwen',0):.0f}s | load_clip {tot.get('load_clip_model',0):.0f}s | load_cache {tot.get('load_ingest_cache',0):.0f}s")
print(f"  qwen_checks {tot.get('qwen_checks',0):.0f}s | sum of QUERY TOTAL {tot.get('query_total',0):.0f}s")
print(f"  tracks sent to Qwen (all videos): {sent} | MATCH total: {match}")
print(f"  last log written: {datetime.datetime.fromtimestamp(max(mt)):%H:%M:%S}")
