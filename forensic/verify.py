"""Stage 2b - Qwen2-VL: (1) plans the yes/no checks from the one-line query, (2) answers them on the best crops
of every shortlisted track. Every answer is Yes / No / Unclear; the weakest answer of a crop decides."""
import gc
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .config import mmss


class Qwen:
    def __init__(self, cfg):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        kw = dict(torch_dtype=torch.float16)
        if cfg["qwen_4bit"]:                                  # one small GPU: 4-bit weights
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            kw["device_map"] = {"": 0}
        else:
            kw["device_map"] = "auto"                         # spread over all visible GPUs
            if cfg["qwen_max_memory"]:
                kw["max_memory"] = {int(k): v for k, v in cfg["qwen_max_memory"].items()}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t0 = time.time()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(cfg["qwen_model"], **kw).eval()
        self.proc = AutoProcessor.from_pretrained(cfg["qwen_model"], min_pixels=cfg["qwen_min_pixels"],
                                                  max_pixels=cfg["qwen_max_pixels"])
        self.tok = self.proc.tokenizer
        self.yes_ids = self._ids(["Yes", "yes", " Yes", " yes"])
        self.no_ids = self._ids(["No", "no", " No", " no"])
        unc = self._ids(["Unclear", " Unclear", "unclear", " unclear"])
        overlap = set(unc) & (set(self.yes_ids) | set(self.no_ids))
        self.unc_ids = [i for i in unc if i not in overlap]
        if not self.unc_ids:
            raise RuntimeError("Could not find a token for 'Unclear' in this Qwen tokenizer.")
        print(f"Qwen loaded in {time.time() - t0:.0f}s; answer token ids yes: {self.yes_ids} no: {self.no_ids} unclear: {self.unc_ids}")

    def _ids(self, words):
        out = set()
        for w in words:
            ids = self.tok.encode(w, add_special_tokens=False)
            if ids:
                out.add(ids[0])
        return sorted(out)

    @torch.no_grad()
    def text(self, prompt, max_new_tokens=450):
        text = self.proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                                             tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    @staticmethod
    def _to3(q):
        q = q.strip()
        base = q[:-len("Answer Yes or No.")].rstrip() if q.lower().endswith("answer yes or no.") else q
        return base + " Answer Yes, No or Unclear. Answer Unclear if the detail is not clearly visible in the image."

    @torch.no_grad()
    def p3(self, images, question):
        """Yes / No / Unclear probabilities (softmax over the three answer tokens): (p_yes, p_no, p_unclear)."""
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": self._to3(question)}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=images, return_tensors="pt").to(self.model.device)
        logits = self.model(**inp).logits[0, -1].float()
        l = torch.stack([torch.logsumexp(logits[self.yes_ids], 0), torch.logsumexp(logits[self.no_ids], 0),
                         torch.logsumexp(logits[self.unc_ids], 0)])
        pr = torch.softmax(l, 0).tolist()
        return pr[0], pr[1], pr[2]


def plan_query(qwen, desc, target):
    """Qwen writes the yes/no checks from the user's one-line description (only details really written in it).
    Returns (attr, raw_reply, note); attr = list of (name, question, crop view f/u/m)."""
    who = "person" if target == "person" else "vehicle"
    start = "Is the person in this image ... / Does the person in this image ..." if target == "person" else "Is this vehicle ... / Does this vehicle ..."
    views = ('"u" = head, hat, upper body, shirt;  "f" = whole body or lower body (trousers, shoes, bags, skin, tattoos on arms or legs);  "m" = text or a logo on the middle of the body'
             if target == "person" else
             '"f" = the whole vehicle (colour, type);  "m" = lettering, a logo or a number plate on the vehicle')
    prompt = (f'You help a CCTV search system. The user describes a {who}: "{desc}".\n'
              f'Reply with ONE JSON object and nothing else: {{"questions": [...]}} with 1 to 4 items, ONE per visual detail that the description '
              f'explicitly states. NEVER add a detail that is not written in the description (no extra colours, types or logos).\n'
              f'Each item is {{"detail": the exact words copied from the description, "name": short_snake_case, '
              f'"question": a yes/no question about ONE image of a {who} (start like: {start}) ending with "Answer Yes or No.", '
              f'"view": one of "u", "f", "m"}}. View meanings: {views}.')
    raw = qwen.text(prompt)
    plan, note = None, ""
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        plan = json.loads(m.group(0))
    except Exception as e:
        note = f"could not parse Qwen's plan ({e})"
    norm = lambda t: re.sub(r"\W+", " ", str(t).lower()).strip()
    dn = norm(desc)
    attr, dropped = [], []
    for it in (plan or {}).get("questions", [])[:8]:
        try:
            det, qn = norm(it.get("detail", "")), str(it["question"]).strip()
            if not det or det not in dn:                      # keep only details that really are in the description
                dropped.append(str(it.get("name", it.get("detail", "?"))))
                continue
            if not qn.lower().rstrip().endswith("answer yes or no."):
                qn += " Answer Yes or No."
            vw = it.get("view", "f")
            attr.append((re.sub(r"\W+", "_", str(it.get("name", det)).strip().lower()).strip("_")[:20] or f"attr{len(attr) + 1}",
                         qn, vw if vw in ("f", "u", "m") else "f"))
        except Exception:
            continue
    # ---- COVERAGE: every part of the user's short prompt must be checked (Qwen often skips one, e.g. "small kid") ----
    split = r"\b(?:with|wearing|carrying|holding|having|has|and|in)\b|[,;]"      # NOT on/at/of/the: "stripes on the left side" stays one phrase
    generic = {"person", "people", "vehicle", "someone", "something", "thing"}
    stem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") else w
    chunks = [re.sub(r"^(?:(?:a|an|the|his|her|their)\s+)+", "", c.strip()) for c in re.split(split, desc.lower()) if c and c.strip()]
    chunks = [c for c in chunks if re.sub(r"\W+", " ", c).strip() and re.sub(r"\W+", " ", c).split()[-1] not in generic]
    qtext = " ".join(norm(q) for _, q, _ in attr)
    qwords = {stem(w) for w in qtext.split()}
    added = []
    for ch in chunks:
        head = stem(norm(ch).split()[-1])                      # head word of the phrase: "small kid" -> kid, "green hat" -> hat
        if head in qwords:
            continue
        q_new = None
        try:
            r2 = qwen.text(f'A CCTV search looks for a {who}: "{desc}". Write ONE yes/no question about ONE image of a {who} asking only whether '
                           f'this part is true: "{ch}". Start like: {start}. End with "Answer Yes or No.". Reply with the question only.',
                           max_new_tokens=60).strip().splitlines()[0]
            if head in {stem(w) for w in norm(r2).split()}:
                q_new = r2 if r2.lower().rstrip().endswith("answer yes or no.") else r2 + " Answer Yes or No."
        except Exception:
            pass
        if not q_new:                                          # template fallback
            kinds = ("kid", "child", "man", "woman", "boy", "girl", "baby", "driver", "car", "truck", "van", "bus", "motorcycle", "bike", "suv", "taxi", "lorry", "minibus")
            art = "" if (" on " in f" {ch} " or " at " in f" {ch} " or ch.endswith("s")) else "a "
            q_new = (f"Is the {who} in this image a {ch}? Answer Yes or No." if head in kinds
                     else f"Does the {who} in this image have {art}{ch}? Answer Yes or No." if target == "person"
                     else f"Does this vehicle have {art}{ch}? Answer Yes or No.")
        attr.append((re.sub(r"\W+", "_", norm(ch)).strip("_")[:20] or "part", q_new, "f"))
        added.append(ch)
        qwords |= {stem(w) for w in norm(q_new).split()}
    if added:
        note = (note + "; " if note else "") + "added checks Qwen skipped: " + ", ".join(added)
    attr = attr[:5]
    if dropped:
        note = (note + "; " if note else "") + "dropped invented checks (not in your description): " + ", ".join(dropped)
    if not attr:                                               # fallback: one generic check straight from the description
        attr = [("match", f"Does this image show {desc}? Answer Yes or No.", "f")]
        note = (note + "; " if note else "") + "no usable questions -> one generic question"
    return attr, raw, note


def plan_and_save(qwen, q, resq):
    """Plan the checks for query `q`, print them, save plan.json. Returns (attr, same_question)."""
    attr, raw, note = plan_query(qwen, q["text"], q["target"])
    print("\n=== QUERY PLANNED BY QWEN (no negative prompts are used) ===")
    print("POS (yours)  :", q["text"])
    print("ATTR (Qwen)  :")
    for n, qu, v in attr:
        print(f"   - [{v}] {n}: {qu}")
    if note:
        print("NOTE:", note)
    with open(os.path.join(resq, "plan.json"), "w") as f:
        json.dump(dict(pos=q["text"], attr=attr, raw=raw, note=note), f, indent=1)
    who = q["target"]
    same_q = (f"The first image is a reference photo of a {who} ({q['text']}). "
              f"Is the {who} in the second image the same one? Answer Yes or No.")
    return attr, same_q


def check_tracks(qwen, cache, cfg, TOP, attr, same_q, ref_pil):
    """Run the checks on the best crops of every shortlisted track. Returns QRES: track key -> result dict."""
    rows = cache.rows

    def crop_img(k, suf):
        return Image.open(os.path.join(cache.crops, f"{rows[k]['id']:06d}_{suf}.jpg")).convert("RGB")

    print("SAME (reference photo) check:", "ON" if ref_pil is not None else "OFF (text-only)")
    t0 = time.time()
    QRES = {}
    for tr in tqdm(TOP, desc="Qwen checks", unit="track", dynamic_ncols=True):
        frames = []
        for k in tr["crops"]:
            a, un, y3, no_ = {}, {}, {}, {}
            for name, question, view in attr:
                py, pn, pu = qwen.p3([crop_img(k, view)], question)
                y3[name], no_[name], un[name] = py, pn, pu                   # raw 3-way probabilities (used for the veto label)
                a[name] = py / (py + pn + 1e-9)                              # yes-vs-no score
            if ref_pil is not None:
                sp = qwen.p3([ref_pil, crop_img(k, "f")], same_q)
                same = sp[0] / (sp[0] + sp[1] + 1e-9)
            else:
                same = None
            attrs = float(np.mean(list(a.values())))
            q_avg = 0.5 * attrs + 0.5 * same if same is not None else attrs  # averaged score (kept for reference)
            q_min = min(list(a.values()) + ([same] if same is not None else []))   # STRICT: the weakest answer decides
            lab_attr = [("yes", "no", "unclear")[int(np.argmax([y3[n_], no_[n_], un[n_]]))] for n_ in a]
            lab = "no" if "no" in lab_attr else "unclear" if "unclear" in lab_attr else "yes"   # frame label from its weakest answer
            qd = q_min if lab == "yes" else 0.0      # DECISION score: a crop where "unclear" (or "no") is the top answer can NOT confirm the track
            frames.append(dict(k=k, attrs=a, unclear=un, same=same, q_avg=q_avg, q=q_min, qd=qd, lab=lab))
        best = max(frames, key=lambda d: (d["qd"], d["q"]))
        QRES[tr["key"]] = dict(frames=frames, q=best["qd"], q_raw=max(d["q"] for d in frames), q_avg=best["q_avg"], best_k=best["k"],
                               mean_q=float(np.mean([d["q"] for d in frames])),
                               any_unclear=any(d["lab"] == "unclear" for d in frames), labs="".join(d["lab"][0] for d in frames))
    print(f"Qwen checked {len(TOP)} tracks x {cfg['n_crops']} frames in {time.time() - t0:.0f}s")

    names = [n for n, _, _ in attr]
    print("\ntrack  seen          " + "  ".join(f"{n[:8]:<8}" for n in names) + "  same    STRICT(ok-only)  (raw)  avg   frames(y/n/u)")
    for tr in TOP[:cfg["print_rows"]]:
        d = QRES[tr["key"]]
        b = max(d["frames"], key=lambda x: (x["qd"], x["q"]))
        same = f"{b['same']:.2f}" if b["same"] is not None else " n/a"
        cols = "  ".join(f"{b['attrs'][n]:<8.2f}" for n in names)
        print(f"{str(tr['key']):>5}  {mmss(tr['t0'])}-{mmss(tr['t1'])}   {cols}  {same}   {d['q']:.2f}      ({d['q_raw']:.2f})   {d['q_avg']:.2f}   {d['labs']}")
    return QRES
