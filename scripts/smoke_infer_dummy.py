# scripts/smoke_infer_dummy.py
"""
Smoke inference on SYNTHETIC data (no dataset required).

- Builds a synthetic (Z,Y,X) volume + binary mask.
- Extracts a k-slice stack at a chosen z.
- Runs UNet2D forward pass (optionally loading a checkpoint).
- Saves an overlay PNG (image + true mask + predicted mask).

This is safe for CI and for reviewers without your private data.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import torch

# --- add src/ to PYTHONPATH (robust to CWD) ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prostate_unet.models.unet2d import UNet2D  # noqa: E402


def resolve_device(device: str) -> torch.device:
    d = (device or "auto").lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if d == "directml":
        if sys.platform != "win32":
            raise RuntimeError("DirectML is only supported on Windows.")
        try:
            import torch_directml  # type: ignore
        except ImportError as e:
            raise RuntimeError("DirectML requested but torch-directml is not installed.") from e
        return torch_directml.device()
    # auto
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def center_crop_or_pad_2d(img: np.ndarray, size: int) -> np.ndarray:
    """Return a (size,size) array: center-crop if larger, zero-pad if smaller."""
    h, w = img.shape
    out = np.zeros((size, size), dtype=img.dtype)

    # crop window in source
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)

    crop = img[y0:y1, x0:x1]

    # paste window in destination
    ph = (size - crop.shape[0]) // 2
    pw = (size - crop.shape[1]) // 2
    out[ph:ph + crop.shape[0], pw:pw + crop.shape[1]] = crop
    return out


def build_k_stack(vol_zyx: np.ndarray, z: int, k: int, crop_size: int) -> np.ndarray:
    """
    vol_zyx: (Z,Y,X)
    return: (k, crop_size, crop_size)
    """
    assert k % 2 == 1, "k must be odd"
    Z = vol_zyx.shape[0]
    half = k // 2
    zs = [min(max(zz, 0), Z - 1) for zz in range(z - half, z + half + 1)]
    stack = np.stack([center_crop_or_pad_2d(vol_zyx[zz], crop_size) for zz in zs], axis=0)
    return stack.astype(np.float32)


def dice_np(pred: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    true = (true > 0).astype(np.uint8)
    inter = float((pred & true).sum())
    denom = float(pred.sum() + true.sum())
    return (2.0 * inter + eps) / (denom + eps)


def save_overlay_png(img: np.ndarray, m_true: np.ndarray, m_pred: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    im = img.astype(np.float32)
    im = (im - im.min()) / (im.max() - im.min() + 1e-6)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(im, cmap="gray")
    ax.imshow(np.ma.masked_where(m_true == 0, m_true), alpha=0.35)
    ax.imshow(np.ma.masked_where(m_pred == 0, m_pred), alpha=0.35)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_dummy_volume(z: int, y: int, x: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic volume (Z,Y,X) and a spherical-ish binary mask.
    """
    rng = np.random.default_rng(seed)
    vol = rng.normal(0, 1, size=(z, y, x)).astype(np.float32)

    zz, yy, xx = np.ogrid[:z, :y, :x]
    cz, cy, cx = z // 2, y // 2, x // 2
    r = min(z, y, x) // 5
    mask = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.uint8)

    vol = vol + 1.25 * mask.astype(np.float32)
    return vol, mask


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke inference on synthetic data (no dataset required).")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "previews" / "smoke_dummy.png"),
                    help="Output overlay PNG path")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "directml"])
    ap.add_argument("--ckpt", default="", help="Optional checkpoint path (state_dict .pth)")
    ap.add_argument("--k-slices", type=int, default=1, help="Odd number of slices to stack as channels (1/3/5/...)")
    ap.add_argument("--base-ch", type=int, default=32, help="Base channels for UNet2D")
    ap.add_argument("--crop-size", type=int, default=256, help="Final H=W size")
    ap.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for predicted mask")
    ap.add_argument("--z", type=int, default=-1, help="Slice index; -1 uses slice with max mask area")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic volume")
    args = ap.parse_args()

    if args.k_slices % 2 != 1:
        raise SystemExit("--k-slices must be odd (1,3,5,...)")

    device = resolve_device(args.device)

    # Create synthetic volume bigger than crop-size to exercise crop logic
    Z = max(32, args.k_slices + 8)
    Y = max(args.crop_size, 300)
    X = max(args.crop_size, 300)
    vol, msk = make_dummy_volume(Z, Y, X, seed=args.seed)

    # choose z
    if args.z >= 0:
        z = min(max(args.z, 0), Z - 1)
    else:
        areas = msk.reshape(Z, -1).sum(axis=1)
        z = int(np.argmax(areas))

    x_stack = build_k_stack(vol, z, args.k_slices, args.crop_size)
    y_true  = center_crop_or_pad_2d(msk[z].astype(np.uint8), args.crop_size)

    # normalize stack
    mu = float(x_stack.mean())
    sd = float(x_stack.std() + 1e-6)
    x_stack = (x_stack - mu) / sd

    x = torch.from_numpy(x_stack)[None, ...].to(device)  # (1,C,H,W)

    model = UNet2D(in_ch=args.k_slices, base=args.base_ch).to(device)
    model.eval()

    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)

    with torch.no_grad():
        logits = model(x)  # (1,1,H,W)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        y_pred = (prob >= args.threshold).astype(np.uint8)

    d = dice_np(y_pred, y_true)
    out_path = Path(args.out)
    save_overlay_png(img=x_stack[args.k_slices // 2], m_true=y_true, m_pred=y_pred, out_path=out_path)

    print(f"[ok] wrote {out_path}")
    print(f"device={device} | ckpt={'(none)' if not args.ckpt else args.ckpt}")
    print(f"k={args.k_slices} base={args.base_ch} crop={args.crop_size} z={z} dice={d:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
