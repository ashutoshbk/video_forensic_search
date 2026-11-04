"""Stage 1b - OpenCLIP vectors for every crop (3 views per detection), resumable in chunks.
Also holds the text/image encoders reused by the search stage."""
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import mmss

GREY = (124, 116, 104)
VIEWS = [("full", "f"), ("upper", "u"), ("mid", "m")]


def load_square(path_or_img):
    """Pad to a square with grey so OpenCLIP's centre-crop does not cut off the edges."""
    im = Image.open(path_or_img).convert("RGB") if isinstance(path_or_img, str) else path_or_img.convert("RGB")
    s = max(im.size)
    return ImageOps.pad(im, (s, s), color=GREY, centering=(0.5, 0.5))


class _CropDS(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.preprocess(load_square(self.paths[i]))
        except Exception:                      # an unreadable crop gets a neutral grey image instead of a crash
            return self.preprocess(Image.new("RGB", (224, 224), GREY))


class ClipEncoder:
    def __init__(self, arch, weights, device, batch_size=128, num_workers=4):
        import open_clip
        self.device, self.batch_size, self.num_workers = device, batch_size, num_workers
        self.fp16 = str(device).startswith("cuda")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=weights, precision="fp16" if self.fp16 else "fp32", device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)

    def _to_device(self, batch):
        return batch.half().to(self.device, non_blocking=True) if self.fp16 else batch.to(self.device)

    @torch.no_grad()
    def _encode(self, batch):
        return F.normalize(self.model.encode_image(self._to_device(batch)).float(), dim=-1).cpu().numpy()

    def embed_paths(self, paths, bar=None):
        dl = DataLoader(_CropDS(paths, self.preprocess), batch_size=self.batch_size, shuffle=False,
                        num_workers=self.num_workers, pin_memory=self.fp16)
        outs = []
        for batch in dl:
            outs.append(self._encode(batch))
            if bar is not None:
                bar.update(int(batch.shape[0]))
        return np.concatenate(outs)

    def embed_images(self, imgs, bs=64):
        """Embed a short list of image paths / PIL images (used for the reference photo)."""
        outs = []
        for s in range(0, len(imgs), bs):
            batch = torch.stack([self.preprocess(load_square(p)) for p in imgs[s:s + bs]])
            outs.append(self._encode(batch))
        return np.concatenate(outs)

    @torch.no_grad()
    def text_vec(self, prompts):
        t = F.normalize(self.model.encode_text(self.tokenizer(prompts).to(self.device)).float(), dim=-1).mean(0)
        return (t / t.norm()).cpu().numpy()


def embed_all_crops(clip, rows, dirs, cfg):
    """Return (Ef, Eu, Em): vectors of the full / upper / middle crop of every row, in row order.
    Every finished chunk is saved, so a stop can be resumed; the finished result is cached as one .npz."""
    out, crops, chunk = dirs["out"], dirs["crops"], cfg["emb_chunk"]
    emb_file = os.path.join(out, f"emb_{cfg['clip_arch']}_{cfg['clip_weights']}_padv1.npz")
    parts_dir = os.path.join(out, f"emb_parts_{len(rows)}")
    if os.path.exists(emb_file):
        z = np.load(emb_file)
        if z["full"].shape[0] == len(rows):
            print("loaded cached vectors", z["full"].shape)
            return z["full"], z["upper"], z["mid"]

    t0 = time.time()
    os.makedirs(parts_dir, exist_ok=True)
    N, res = len(rows), {}
    bar = tqdm(total=3 * N, desc="2/2 embed", unit="crop", dynamic_ncols=True, smoothing=0.05)
    for name, suf in VIEWS:
        mats = []
        for ci, s in enumerate(range(0, N, chunk)):
            n_this = min(chunk, N - s)
            part = os.path.join(parts_dir, f"{name}_{ci:04d}.npy")
            if os.path.exists(part):
                m = np.load(part)
                if m.shape[0] == n_this:
                    mats.append(m)
                    bar.update(n_this)
                    continue
            bar.set_description(f"2/2 embed [{name} view]")
            paths = [os.path.join(crops, f"{r['id']:06d}_{suf}.jpg") for r in rows[s:s + n_this]]
            m = clip.embed_paths(paths, bar)
            tmp = part[:-4] + ".tmp.npy"
            np.save(tmp, m)
            os.replace(tmp, part)
            mats.append(m)
        res[name] = np.concatenate(mats)
    bar.close()
    np.savez(emb_file, full=res["full"], upper=res["upper"], mid=res["mid"])
    shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"embedded {N} x 3 crops -> {res['full'].shape}, {mmss(time.time() - t0)}")
    return res["full"], res["upper"], res["mid"]


def load_embeddings(dirs, cfg, n_rows):
    """Load the vectors written by embed_all_crops (used by the search stage)."""
    emb_file = os.path.join(dirs["out"], f"emb_{cfg['clip_arch']}_{cfg['clip_weights']}_padv1.npz")
    if not os.path.exists(emb_file):
        raise RuntimeError(f"no vectors found ({emb_file}): run `python run_ingest.py` first")
    z = np.load(emb_file)
    if z["full"].shape[0] != n_rows:
        raise RuntimeError("the saved vectors do not match the detections: run `python run_ingest.py` again")
    return z["full"], z["upper"], z["mid"]
