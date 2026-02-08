# scripts/smoke_infer_one_case.py
"""
Smoke inference on ONE real NIfTI case (image+label).

- Locates the pair by --image/--label OR by --case + default dirs.
- Converts arrays to a consistent (Z,Y,X) convention (heuristic).
- Extracts a k-slice stack at a chosen slice (default: max mask area).
- Runs UNet2D forward pass (optionally loading a checkpoint).
- Saves an overlay PNG and prints a quick 2D Dice.

This script intentionally avoids SciPy so installs stay lightweight.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import torch

import nibabel as nib

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
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def center_crop_or_pad_2d(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape
    out = np.zeros((size, size), dtype=img.dtype)

    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)

    crop = img[y0:y1, x0:x1]

    ph = (size - crop.shape[0]) // 2
    pw = (size - crop.shape[1]) // 2
    out[ph:ph + crop.shape[0], pw:pw + crop.shape[1]] = crop
    return out


def to_zyx(vol: np.ndarray) -> np.ndarray:
    """
    Heuristic conversion to (Z,Y,X):
      - Assume the slice axis Z is the smallest dimension.
      - If smallest dim is last (X,Y,Z), transpose to (Z,Y,X).
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    ax = int(np.argmin(vol.shape))
    if ax == 0:
        # already (Z,*,*)
        return vol
    if ax == 2:
        # likely (X,Y,Z) -> (Z,Y,X)
        return np.transpose(vol, (2, 1, 0))
    # ax == 1 (rare): (X,Z,Y) or (Y,Z,X) - pick a consistent mapping
    return np.transpose(vol, (1, 2, 0))


def build_k_stack(vol_zyx: np.ndarray, z: int, k: int, crop_size: int) -> np.ndarray:
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


def find_by_case(case: str, mr_dir: Path, label_dir: Path) -> tuple[Path, Path]:
    mr_candidates = sorted(list(mr_dir.glob(f"{case}*.nii*")))
    lb_candidates = sorted(list(label_dir.glob(f"{case}*.nii*")))
    if not mr_candidates:
        raise FileNotFoundError(f"No MR files found for case '{case}' in {mr_dir}")
    if not lb_candidates:
        raise FileNotFoundError(f"No label files found for case '{case}' in {label_dir}")
    return mr_candidates[0], lb_candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke inference on one real case (NIfTI).")
    ap.add_argument("--image", default="", help="Path to MR .nii/.nii.gz (optional if using --case)")
    ap.add_argument("--label", default="", help="Path to label .nii/.nii.gz (optional if using --case)")
    ap.add_argument("--case", default="", help="Case prefix (e.g. B012_Week1) to find files in default dirs")
    ap.add_argument("--mr-dir", default=str(ROOT / "semantic_MRs"), help="Directory containing MR NIfTIs")
    ap.add_argument("--label-dir", default=str(ROOT / "semantic_labels_only"), help="Directory containing label NIfTIs")
    ap.add_argument("--out", default="", help="Output overlay PNG (default: artifacts/previews/smoke_<case>.png)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "directml"])
    ap.add_argument("--ckpt", default="", help="Optional checkpoint path (state_dict .pth)")
    ap.add_argument("--k-slices", type=int, default=1, help="Odd number of slices to stack as channels")
    ap.add_argument("--base-ch", type=int, default=32, help="Base channels for UNet2D")
    ap.add_argument("--crop-size", type=int, default=256, help="Final H=W size")
    ap.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for predicted mask")
    ap.add_argument("--z", type=int, default=-1, help="Slice index; -1 uses slice with max mask area")
    args = ap.parse_args()

    if args.k_slices % 2 != 1:
        raise SystemExit("--k-slices must be odd (1,3,5,...)")

    device = resolve_device(args.device)

    if args.image and args.label:
        img_path = Path(args.image)
        lbl_path = Path(args.label)
        case = args.case or img_path.stem.split(".")[0]
    else:
        if not args.case:
            raise SystemExit("Provide either --image/--label OR --case.")
        case = args.case
        img_path, lbl_path = find_by_case(case, Path(args.mr_dir), Path(args.label_dir))

    out_path = Path(args.out) if args.out else (ROOT / "artifacts" / "previews" / f"smoke_{case}.png")

    img_nii = nib.as_closest_canonical(nib.load(str(img_path)))
    lbl_nii = nib.as_closest_canonical(nib.load(str(lbl_path)))

    img = img_nii.get_fdata().astype(np.float32)
    lbl = lbl_nii.get_fdata().astype(np.float32)

    if img.shape != lbl.shape:
        raise RuntimeError(f"Image/label shape mismatch: img={img.shape}, lbl={lbl.shape}")

    img_zyx = to_zyx(img)
    lbl_zyx = to_zyx(lbl)

    Z = img_zyx.shape[0]
    mask_full = (lbl_zyx > 0).astype(np.uint8)

    # choose z
    if args.z >= 0:
        z = min(max(args.z, 0), Z - 1)
    else:
        areas = mask_full.reshape(Z, -1).sum(axis=1)
        z = int(np.argmax(areas))

    x_stack = build_k_stack(img_zyx, z, args.k_slices, args.crop_size)
    y_true = center_crop_or_pad_2d(mask_full[z], args.crop_size)

    # normalize stack (z-score)
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
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        y_pred = (prob >= args.threshold).astype(np.uint8)

    d = dice_np(y_pred, y_true)
    save_overlay_png(img=x_stack[args.k_slices // 2], m_true=y_true, m_pred=y_pred, out_path=out_path)

    print(f"[ok] img={img_path.name} lbl={lbl_path.name}")
    print(f"[ok] wrote {out_path}")
    print(f"device={device} | ckpt={'(none)' if not args.ckpt else args.ckpt}")
    print(f"k={args.k_slices} base={args.base_ch} crop={args.crop_size} z={z} dice={d:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
