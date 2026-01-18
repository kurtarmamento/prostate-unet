# Purpose: Pick top/bottom-N cases from metrics_<split>.csv and save overlay PNGs.
# Usage (PowerShell):
#   python scripts\inspect_cases.py `
#     --split test `
#     --metrics artifacts\preds_test\metrics_test.csv `
#     --preds-dir artifacts\preds_test `
#     --splits-csv artifacts\cache_splits.csv `
#     --threshold 0.30 `
#     --top 2 --bottom 2 `
#     --outdir artifacts\previews

from __future__ import annotations
from pathlib import Path
import csv, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_prob_and_cache(pred_path: Path, cache_path: Path):
    pr = np.load(pred_path)["prob"]        # (Zc,Hc,Wc) float32
    cz = np.load(cache_path)
    img = cz["image"].astype(np.float32)   # (Zc,Hc,Wc) z-scored
    gt  = cz["label"].astype(np.uint8)     # (Zc,Hc,Wc) 0/1
    return pr, img, gt

def overlay(ax, img2d, mask2d, title):
    # Show the base slice (intensity); then alpha-blend a mask
    ax.imshow(img2d, cmap="gray")
    # Red = predicted; Green = ground truth (kept simple & distinct)
    ax.imshow(np.ma.masked_where(mask2d==0, mask2d), alpha=0.35)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def make_preview(pred_path: Path, cache_path: Path, out_png: Path, threshold: float):
    pr, img, gt = load_prob_and_cache(pred_path, cache_path)
    z = pr.shape[0] // 2
    pred2d = (pr[z] >= threshold).astype(np.uint8)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    overlay(axs[0], img[z], np.zeros_like(pred2d), "Image")
    overlay(axs[1], img[z], pred2d, f"Pred @t={threshold:.2f}")
    overlay(axs[2], img[z], gt[z], "Ground Truth")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--metrics", required=True)          # artifacts/preds_<split>/metrics_<split>.csv
    ap.add_argument("--preds-dir", required=True)        # artifacts/preds_<split>
    ap.add_argument("--splits-csv", required=True)       # artifacts/cache_splits.csv
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--bottom", type=int, default=2)
    ap.add_argument("--outdir", default="artifacts/previews")
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    outdir = Path(args.outdir)

    # Read metrics and sort by best_dice if available, else dice@t
    with open(args.metrics, newline="") as f:
        rows = list(csv.DictReader(f))
    def score(r):
        return float(r.get("best_dice") or r.get("dice@t") or 0.0)
    rows_sorted = sorted(rows, key=score, reverse=True)

    # Map subject -> cache_path from cache_splits.csv
    with open(args.splits_csv, newline="") as f:
        split_rows = list(csv.DictReader(f))
    cache_map = {r["subject_week"]: r["cache_path"] for r in split_rows if r.get("split", args.split)==args.split}

    # Select subjects
    picks = rows_sorted[:args.top] + rows_sorted[-args.bottom:] if args.bottom > 0 else rows_sorted[:args.top]
    for r in picks:
        subj = r["subject_week"]
        pred_path = preds_dir / f"{subj}_pred.npz"
        cache_path = Path(cache_map.get(subj, ""))
        if not pred_path.exists() or not cache_path.exists():
            print(f"[skip] {subj}: missing {pred_path} or {cache_path}")
            continue
        png = outdir / f"{args.split}_{subj}.png"
        print(f"[make] {png.name}")
        make_preview(pred_path, cache_path, png, args.threshold)

if __name__ == "__main__":
    main()
