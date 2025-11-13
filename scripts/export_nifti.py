"""
Export predicted masks (from saved probs) back to full-size NIfTI in the
preprocessed RAS/resampled space (same grid/affine as before crop).
Requires that each cache npz includes: affine, orig_shape, crop_slices.
"""

from __future__ import annotations
from pathlib import Path
import sys, csv, argparse
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]

def keep_lcc_3d(bin3d: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label
        lab, n = label(bin3d)
        if n == 0: return bin3d
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        return (lab == sizes.argmax()).astype(np.uint8)
    except Exception:
        return bin3d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-csv", default=str(ROOT / "artifacts" / "cache_splits.csv"))
    ap.add_argument("--preds-dir",  default=str(ROOT / "artifacts" / "preds_val"))
    ap.add_argument("--split",      default="val")
    ap.add_argument("--threshold",  type=float, default=0.5)
    ap.add_argument("--lcc",        action="store_true")
    ap.add_argument("--outdir",     default=str(ROOT / "artifacts" / "nifti_out"))
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.splits_csv).open()))
    rows = [r for r in rows if r.get("split","val")==args.split]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for r in rows:
        subj = r.get("subject_week", Path(r["cache_path"]).stem)
        npz_pred = Path(args.preds_dir) / f"{subj}_pred.npz"
        if not npz_pred.exists():
            print(f"skip {subj}: no probs"); continue

        pr = np.load(npz_pred)
        probs = pr["prob"]                            # (Zc,Hc,Wc) cropped space
        cache_path = pr["cache_path"].item() if "cache_path" in pr.files else r["cache_path"]

        meta = np.load(cache_path)
        affine = meta["affine"]                       # 4x4
        orig_shape = tuple(int(x) for x in meta["orig_shape"])
        z0, z1, y0, y1, x0, x1 = (int(v) for v in meta["crop_slices"])

        # threshold (+ optional LCC) in crop space
        mask_crop = (probs >= args.threshold).astype(np.uint8)
        if args.lcc:
            mask_crop = keep_lcc_3d(mask_crop)

        # paste crop back into full preprocessed space
        full = np.zeros(orig_shape, dtype=np.uint8)
        full[z0:z1, y0:y1, x0:x1] = mask_crop  # (Zc,Hc,Wc) -> (Z,H,W)

        # write NIfTI
        out_path = outdir / f"{subj}_mask_t{args.threshold:.2f}.nii.gz"
        nib.Nifti1Image(full.astype(np.uint8), affine).to_filename(str(out_path))
        print("wrote", out_path)

if __name__ == "__main__":
    main()
