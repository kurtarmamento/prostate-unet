"""
Full-volume inference + per-volume metrics.
- Runs the trained model over a split (val/test).
- Saves per-volume probability maps (.npz) and a metrics CSV.
- Reports best global threshold on that split.
- Also embeds subject + cache_path inside each .npz for later export.
"""

from __future__ import annotations
from pathlib import Path
import sys, csv, json, argparse
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prostate_unet.models.unet2d import UNet2D

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", True
    try:
        import torch_directml  # noqa: F401
        dml = torch_directml.device()
        return dml, "dml", False
    except Exception:
        pass
    return torch.device("cpu"), "cpu", False

def keep_lcc_3d(bin3d: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label
        lab, n = label(bin3d)
        if n == 0: return bin3d
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        return (lab == sizes.argmax()).astype(np.uint8)
    except Exception:
        return bin3d

@torch.no_grad()
def infer_volume(model: torch.nn.Module, vol: np.ndarray, crop: int, k: int, device, use_amp: bool) -> np.ndarray:
    """Return per-voxel probability map (Z,H,W) in cached-crop space."""
    Z, h, w = vol.shape
    H = W = crop
    preds = []
    for z in range(Z):
        # 2.5D context
        if k == 1:
            sl = vol[z:z+1]
        else:
            half = k // 2
            zs = [min(max(zz, 0), Z - 1) for zz in range(z - half, z + half + 1)]
            sl = vol[zs]

        # center-crop if larger
        y0 = max(0, (h - H) // 2); x0 = max(0, (w - W) // 2)
        sl_crop = sl[:, y0:y0+H, x0:x0+W]

        # pad if smaller
        if sl_crop.shape[1] < H or sl_crop.shape[2] < W:
            C, hh, ww = sl_crop.shape
            ph = max(0, H - hh); pw = max(0, W - ww)
            top = ph // 2; bottom = ph - top
            left = pw // 2; right = pw - left
            sl_crop = np.pad(sl_crop, ((0,0),(top,bottom),(left,right)), mode="edge")

        # forward
        x = torch.from_numpy(sl_crop[None, ...]).to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                p = torch.sigmoid(model(x)).float().cpu().numpy()[0, 0]
        else:
            p = torch.sigmoid(model(x)).float().cpu().numpy()[0, 0]

        # paste-back to (h,w)
        pad_y = max(0, (H - h) // 2); pad_x = max(0, (W - w) // 2)
        if h >= H:
            y_src0, y_src1 = 0, H; y_dst0, y_dst1 = y0, y0 + H
        else:
            y_src0, y_src1 = pad_y, pad_y + h; y_dst0, y_dst1 = 0, h
        if w >= W:
            x_src0, x_src1 = 0, W; x_dst0, x_dst1 = x0, x0 + W
        else:
            x_src0, x_src1 = pad_x, pad_x + w; x_dst0, x_dst1 = 0, w

        canvas = np.zeros((h, w), dtype=np.float32)
        canvas[y_dst0:y_dst1, x_dst0:x_dst1] = p[y_src0:y_src1, x_src0:x_src1]
        preds.append(canvas)

    return np.stack(preds, axis=0)

def dice_3d(a: np.ndarray, b: np.ndarray, eps=1e-6) -> float:
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter) / (denom + eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-csv", default=str(ROOT / "artifacts" / "cache_splits.csv"))
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", default=str(ROOT / "artifacts" / "preds_val"))
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--k-slices", type=int, default=5)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--lcc", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device, devtype, use_amp = pick_device()
    print(f"[device] {devtype}  AMP={'on' if use_amp else 'off'}")

    model = UNet2D(in_ch=args.k_slices, base=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    rows = list(csv.DictReader(Path(args.splits_csv).open()))
    rows = [r for r in rows if r.get("split", "val") == args.split]
    if not rows:
        print(f"No rows for split={args.split}"); return

    cand_ts = (0.3, 0.4, 0.5, 0.6)
    sum_dice_per_t = {t: 0.0 for t in cand_ts}; count = 0

    csv_path = outdir / f"metrics_{args.split}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_week","pred_path","vox_pred","vox_gt","dice@t","best_dice","best_t"])

        for r in rows:
            d = np.load(r["cache_path"])
            vol = d["image"].astype(np.float32)
            msk = d["label"].astype(np.uint8)
            probs = infer_volume(model, vol, args.crop_size, args.k_slices, device, use_amp)

            # fixed threshold
            pred = (probs >= args.threshold).astype(np.uint8)
            if args.lcc: pred = keep_lcc_3d(pred)
            dice_fixed = dice_3d(pred, msk)

            # sweep
            best_d, best_t = 0.0, args.threshold
            for t in cand_ts:
                pb = (probs >= t).astype(np.uint8)
                if args.lcc: pb = keep_lcc_3d(pb)
                dsc = dice_3d(pb, msk)
                if dsc > best_d:
                    best_d, best_t = dsc, t
                sum_dice_per_t[t] += dsc
            count += 1

            subj = r.get("subject_week", Path(r["cache_path"]).stem)
            pred_path = outdir / f"{subj}_pred.npz"
            # Save probs + meta for later export
            np.savez_compressed(pred_path,
                                prob=probs.astype(np.float32),
                                subject=str(subj),
                                cache_path=r["cache_path"])

            w.writerow([subj, str(pred_path), int(pred.sum()), int(msk.sum()),
                        f"{dice_fixed:.4f}", f"{best_d:.4f}", f"{best_t:.2f}"])

    # best global threshold on this split
    best_global_t, best_mean = None, -1.0
    for t in cand_ts:
        mean_d = (sum_dice_per_t[t] / max(1, count))
        if mean_d > best_mean:
            best_mean, best_global_t = mean_d, t

    report = {
        "split": args.split,
        "num_vols": count,
        "threshold_fixed": args.threshold,
        "best_global_threshold": best_global_t,
        "best_global_mean_dice": round(float(best_mean), 4),
        "lcc": bool(args.lcc)
    }
    (outdir / f"summary_{args.split}.json").write_text(json.dumps(report, indent=2))
    print("Done:", json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
