"""Reference photo: embed it (plus a mirrored copy, the top 60% and the centre) into one OpenCLIP vector."""
import os

import numpy as np
from PIL import Image, ImageOps


def load_reference(path, clip, require=False):
    """Return (ref_pil, ref_vec). Both are None when there is no photo (text-only search).
    require=True stops with an error if a photo was asked for but is missing."""
    if path is None:
        if require:
            raise ValueError("require_reference is on but no `reference` photo was given")
        print("no reference photo -> text-only search")
        return None, None
    if not os.path.exists(path):
        msg = f"reference photo '{path}' NOT FOUND"
        if require:
            raise FileNotFoundError(msg)
        print(msg, "-> text-only search")
        return None, None
    ref_pil = Image.open(path).convert("RGB")
    w, h = ref_pil.size
    top60 = ref_pil.crop((0, 0, w, int(0.6 * h)))
    mid = ref_pil.crop((int(0.15 * w), int(0.15 * h), int(0.85 * w), int(0.85 * h)))
    views = [ref_pil, ImageOps.mirror(ref_pil), top60, ImageOps.mirror(top60), mid]
    rv = clip.embed_images(views).mean(0)
    print("reference photo FOUND and embedded:", path, ref_pil.size)
    return ref_pil, rv / np.linalg.norm(rv)
