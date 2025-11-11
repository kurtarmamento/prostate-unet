# scripts/quick_overlay.py
# Render a single overlay PNG with MULTI-LABEL coloring.
# Each label value gets its own color (filled or contour). Robust to CWD.

from pathlib import Path
import argparse, csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
from matplotlib.patches import Patch

# --- locate artifacts/ robustly (scripts/ and artifacts/ are siblings) ---
HERE = Path(__file__).resolve().parent           # .../prostate-unet/scripts
ROOT = HERE.parent                                # .../prostate-unet
ARTIFACTS = ROOT / "artifacts"
(ARTIFACTS / "examples").mkdir(parents=True, exist_ok=True)

def resolve_from_csv(pth: str, csv_path: Path) -> Path:
    """Make absolute path; if relative, resolve against the CSV file's folder."""
    p = Path(pth.strip())
    return p if p.is_absolute() else (csv_path.parent / p).resolve()

def parse_args():
    ap = argparse.ArgumentParser(description="Render one overlay with multi-label coloring.")
    ap.add_argument("--volumes-csv", default=str(ARTIFACTS / "splits_volumes.csv"),
                    help="CSV with columns image_path,label_path (default: artifacts/splits_volumes.csv)")
    ap.add_argument("--row-index", type=int, default=0, help="Row index (0-based) to render")
    ap.add_argument("--z", type=int, default=None,
                    help="Slice index to render; if omitted and --auto-slice, pick slice with max mask area")
    ap.add_argument("--auto-slice", action="store_true",
                    help="Choose the slice with the largest combined mask area if --z not given")
    ap.add_argument("--out", default=str(ARTIFACTS / "examples" / "vol_overlay.png"),
                    help="Output PNG (default: artifacts/examples/vol_overlay.png)")
    # --- multi-label controls ---
    ap.add_argument("--label-values", required=True,
                    help="Comma-separated integer label values to display, e.g. '1' or '1,2,3'")
    ap.add_argument("--colors", default=None,
                    help="Comma-separated colors matching labels (e.g. 'red,#00ff00,tab:blue'). "
                         "If fewer than labels, colors will cycle. If omitted, uses Matplotlib defaults.")
    ap.add_argument("--alpha", type=float, default=0.35, help="Fill transparency (ignored in --edge-only)")
    ap.add_argument("--edge-only", action="store_true", help="Draw contours instead of filled regions")
    ap.add_argument("--legend", action="store_true", help="Add a legend mapping colors to label values")
    ap.add_argument("--inspect", action="store_true",
                    help="Print label histogram for the chosen row and exit")
    return ap.parse_args()

def main():
    args = parse_args()

    # Resolve CSV path relative to project root if needed
    csv_path = Path(args.volumes_csv)
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()
    if not csv_path.exists():
        raise SystemExit(f"[error] CSV not found: {csv_path}")

    # Read selected row
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[error] No rows in {csv_path}")
    if not (0 <= args.row_index < len(rows)):
        raise SystemExit(f"[error] row-index {args.row_index} out of range 0..{len(rows)-1}")
    row = rows[args.row_index]

    img_p = resolve_from_csv(row["image_path"], csv_path)
    lbl_p = resolve_from_csv(row["label_path"], csv_path)
    if not img_p.exists() or not lbl_p.exists():
        raise SystemExit(f"[error] Missing file(s):\n  img={img_p}\n  lbl={lbl_p}")

    # Load volumes (NiBabel handles .nii/.nii.gz)
    img = nib.load(str(img_p)).get_fdata().astype(np.float32)
    lbl = nib.load(str(lbl_p)).get_fdata()

    # Parse label values
    labels = [int(s) for s in args.label_values.split(",")]

    # Optional inspect mode
    if args.inspect:
        vals, counts = np.unique(lbl.astype(np.int64), return_counts=True)
        total = counts.sum()
        print(f"[inspect] {lbl_p.name} — top values:")
        for v, c in sorted(zip(vals, counts), key=lambda t: -t[1])[:30]:
            print(f"  value {int(v):>4}: {c} voxels ({c/total:.2%})")
        print(f"Selected labels: {labels}")
        return

    # Choose slice
    if args.z is not None:
        z = args.z
    elif args.auto_slice:
        # combined area across all chosen labels
        combined = np.isin(lbl, labels)
        z = int(np.argmax(combined.sum(axis=(0, 1))))
    else:
        z = img.shape[2] // 2
    if not (0 <= z < img.shape[2]):
        raise SystemExit(f"[error] z={z} out of range 0..{img.shape[2]-1}")

    # Normalize image for display
    sl_img = img[:, :, z]
    sl_img = (sl_img - sl_img.mean()) / (sl_img.std() + 1e-6)

    # Choose colors
    if args.colors:
        raw_cols = [c.strip() for c in args.colors.split(",") if c.strip()]
    else:
        # Matplotlib default cycle
        raw_cols = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not raw_cols:
            raw_cols = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    cols = [to_rgba(raw_cols[i % len(raw_cols)]) for i in range(len(labels))]

    # Prepare figure
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(sl_img, cmap="gray")

    legend_handles = []

    # Draw each label in its own color
    for lab, col in zip(labels, cols):
        mask = (lbl[:, :, z] == lab)
        if not mask.any():
            continue
        if args.edge_only:
            # contour outline in the label's color
            plt.contour(mask.astype(float), levels=[0.5], colors=[col], linewidths=1)
        else:
            # filled overlay; mask zeros out and color the ones
            plt.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([col]), alpha=args.alpha)
        legend_handles.append(Patch(facecolor=col, edgecolor="none", label=f"label={lab}"))

    if args.legend and legend_handles:
        plt.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8)

    # Save
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[info] wrote {out_path}")

if __name__ == "__main__":
    main()
