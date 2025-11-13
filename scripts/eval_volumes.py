# scripts/eval_volumes.py
"""
Re-evaluate Dice from saved probability maps (no model forward).
- Use this to try different thresholds and/or LCC without re-running inference.
"""

from pathlib import Path
import sys, csv, argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def dice_3d(a: np.ndarray, b: np.ndarray, eps=1e-6) -> float:
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter) / (denom + eps)

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
    ap.add_argument("--splits_csv", default=str(ROOT / "artifacts" / "cache_splits.csv"))
    ap.add_argument("--preds_dir",  default=str(ROOT / "artifacts" / "preds_val"))
    ap.add_argument("--split",      default="val")
    ap.add_argument("--threshold",  type=float, default=0.5)
    ap.add_argument("--lcc",        action="store_true")
    args = ap.parse_args()

    # FIX: use args.splits_csv (underscore), not args.splits-csv (invalid)
    rows = list(csv.DictReader(Path(args.splits_csv).open()))
    rows = [r for r in rows if r.get("split","val")==args.split]
    if not rows:
        print(f"No rows for split={args.split} in {args.splits_csv}")
        return

    dices = []
    for r in rows:
        subj = r.get("subject_week", Path(r["cache_path"]).stem)
        npz = Path(args.preds_dir) / f"{subj}_pred.npz"
        if not npz.exists():
            # Silent skip is fine; you can print if you want visibility:
            # print(f"Missing probs for {subj} at {npz}")
            continue
        pr = np.load(npz)["prob"]
        gt = np.load(r["cache_path"])["label"].astype(np.uint8)
        pred = (pr >= args.threshold).astype(np.uint8)
        if args.lcc:
            pred = keep_lcc_3d(pred)
        dices.append(dice_3d(pred, gt))

    if dices:
        print(f"{args.split} Dice@t={args.threshold}: mean={np.mean(dices):.4f} n={len(dices)}")
    else:
        print("No predictions found in preds_dir; did you run infer_volumes.py?")

if __name__ == "__main__":
    main()
