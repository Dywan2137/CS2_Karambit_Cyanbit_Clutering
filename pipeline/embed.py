"""Feature extractors: DINOv2, CLIP, ResNet50 and a handcrafted colour baseline.

Each extractor maps a list of HxWx3 uint8 blade crops to an (N, D) float array.
`embed_paths` adds test-time augmentation: the embedding of a seed is the mean
over a few slightly shifted copies, so a pattern that is shifted by a few
pixels lands on almost the same vector.
"""

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import config as cfg

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# ---- shared helpers ---------------------------------------------------------

def shift_image(img, dx, dy, fill):
    """Translate by (dx, dy) pixels; uncovered area gets `fill`."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    border = tuple(int(v) for v in np.atleast_1d(fill)) if img.ndim == 3 else int(fill)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(images, wh, mean, std):
    """Resize to wh=(width, height), normalise, return an NCHW tensor."""
    import torch
    x = np.stack([cv2.resize(im, wh, interpolation=cv2.INTER_AREA) for im in images])
    x = (x.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def input_size(name, proc_hw):
    """(width, height) of the image a model is fed, given the processed crop's (h, w).
    Single source of truth, used by the extractors and by save_model_inputs."""
    if name == "dinov2":
        # keep the processed aspect ratio; both sides multiples of the 14 px patch
        h, w = proc_hw
        gw = cfg.DINO_WIDTH // 14
        gh = max(1, int(round(gw * h / w)))
        return gw * 14, gh * 14
    if name in ("clip", "resnet50"):
        return 224, 224
    h, w = proc_hw                       # colorhist works on the processed crop as-is
    return w, h


# ---- extractors -------------------------------------------------------------

class DinoV2:
    """CLS token + mean of patch tokens that fall on the blade."""

    name = "dinov2"

    def __init__(self, mask):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.dev = _device()
        self.model = AutoModel.from_pretrained(cfg.DINO_NAME).to(self.dev).eval()
        self.wh = input_size(self.name, mask.shape)
        gw, gh = self.wh[0] // 14, self.wh[1] // 14
        m = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > 0.5
        self.patch_w = torch.from_numpy(m.flatten().astype(np.float32)).to(self.dev)

    def __call__(self, images):
        x = _to_tensor(images, self.wh, IMAGENET_MEAN, IMAGENET_STD).to(self.dev)
        with self.torch.no_grad(), self.torch.autocast(self.dev, enabled=self.dev == "cuda"):
            h = self.model(pixel_values=x).last_hidden_state.float()
        cls = h[:, 0]
        patches = h[:, -self.patch_w.numel():]               # skip CLS (+ registers)
        w = self.patch_w[None, :, None]
        pooled = (patches * w).sum(1) / w.sum().clamp(min=1)
        return self.torch.cat([cls, pooled], dim=1).cpu().numpy()


class Clip:
    name = "clip"

    def __init__(self, mask):
        import open_clip
        import torch
        self.torch = torch
        self.dev = _device()
        self.model, _, _ = open_clip.create_model_and_transforms(
            cfg.CLIP_ARCH, pretrained=cfg.CLIP_WEIGHTS)
        self.model = self.model.to(self.dev).eval()

    def __call__(self, images):
        x = _to_tensor(images, input_size(self.name, None), CLIP_MEAN, CLIP_STD).to(self.dev)
        with self.torch.no_grad():
            return self.model.encode_image(x).float().cpu().numpy()


class ResNet50:
    name = "resnet50"

    def __init__(self, mask):
        import torch
        import torchvision
        self.torch = torch
        self.dev = _device()
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        self.model = m.to(self.dev).eval()

    def __call__(self, images):
        x = _to_tensor(images, input_size(self.name, None), IMAGENET_MEAN, IMAGENET_STD).to(self.dev)
        with self.torch.no_grad():
            return self.model(x).cpu().numpy()


class ColorHist:
    """HSV histogram of blade pixels + coarse colour layout + residual vs mean image."""

    name = "colorhist"

    def __init__(self, mask, mean_crop=None):
        self.mask = mask
        self.mean_crop = mean_crop
        self.grid = 24

    def _one(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        m = self.mask.astype(np.uint8)
        hist = cv2.calcHist([hsv], [0, 1, 2], m, [18, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        hist /= max(hist.sum(), 1)
        hist = np.sqrt(hist)                                   # Hellinger

        small = cv2.resize(img, (self.grid, self.grid), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.mean_crop is not None:
            small -= cv2.resize(self.mean_crop, (self.grid, self.grid), interpolation=cv2.INTER_AREA)
        sm = cv2.resize(self.mask.astype(np.float32), (self.grid, self.grid),
                        interpolation=cv2.INTER_AREA) > 0.5
        layout = (small[sm] / 255.0).ravel()

        def unit(v):
            return v / max(np.linalg.norm(v), 1e-8)
        return np.concatenate([unit(hist), unit(layout)]).astype(np.float32)

    def __call__(self, images):
        return np.stack([self._one(im) for im in images])


EXTRACTORS = {c.name: c for c in (DinoV2, Clip, ResNet50, ColorHist)}


def build_extractor(name, mask, mean_crop=None):
    if name not in EXTRACTORS:
        raise SystemExit(f"Unknown model {name!r}. Choose from {list(EXTRACTORS)}")
    if name == "colorhist":
        return ColorHist(mask, mean_crop)
    return EXTRACTORS[name](mask)


# ---- running ----------------------------------------------------------------

def embed_images(extractor, images, fill, shifts=cfg.TTA_SHIFTS):
    """Embeddings of in-memory images, averaged over shifted copies."""
    if not shifts:
        shifts = [(0, 0)]
    acc = None
    for fx, fy in shifts:
        batch = images if (fx, fy) == (0, 0) else [
            shift_image(im, fx * im.shape[1], fy * im.shape[0], fill) for im in images]
        f = extractor(batch)
        acc = f if acc is None else acc + f
    return acc / len(shifts)


def embed_paths(extractor, paths, fill, batch_size=cfg.BATCH_SIZE, shifts=cfg.TTA_SHIFTS):
    out = []
    for i in tqdm(range(0, len(paths), batch_size), desc=f"embed {extractor.name}"):
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]
        out.append(embed_images(extractor, imgs, fill, shifts))
    return np.concatenate(out).astype(np.float32)


def embedding_path(name, tta=True):
    return cfg.EMB_DIR / f"{name}{'' if tta else '_notta'}.npy"


def load_embeddings(name, tta=True):
    return np.load(embedding_path(name, tta))


def save_model_inputs(names, paths, seeds, proc_hw, out_root=cfg.MODEL_INPUTS_DIR):
    """Save every processed crop resized exactly as each model receives it
    (before colour normalisation), to out_root/<model>/seed_XXXX.png."""
    for name in names:
        wh = input_size(name, proc_hw)
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        for seed, p in tqdm(list(zip(seeds, paths)), desc=f"{name} {wh[0]}x{wh[1]}"):
            im = np.asarray(Image.open(p).convert("RGB"))
            Image.fromarray(cv2.resize(im, wh, interpolation=cv2.INTER_AREA)).save(
                out / f"seed_{seed:04d}.png")
        print(f"  {name}: {len(seeds)} images at {wh[0]}x{wh[1]} -> {out}")
