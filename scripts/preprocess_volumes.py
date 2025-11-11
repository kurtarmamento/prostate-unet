# Iterate artifacts/splits_volumes.csv, preprocess each pair, and acache .npz files + a manifest

from __future__ import annotations

import sys
from pathlib import Path
import argparse, csv, json
import numpy as np
import nibabel as nib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok = True)


def resolve_from_csv(pth: str, csv_path: Path) -> Path:
    p = Path(pth.strip())
    return p if p.is_absolute() else (csv_path.parent / p).resolve()

def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess volume pairs and cache to .npz")
    ap.add_argument("--volumes-csv", default=str(ARTIFACTS / "splits_volumes.csv"),
                    help="Input CSV with columns image_path,label_path,subject_week")
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "cache"),
                    help="Folder to write .npz files")
    ap.add_argument("--out-manifest", default=str(ARTIFACTS / "cache_volumes.csv"),
                    help="CSV manifest of cached files")
    ap.add_argument("--spacing", default="1.0,1.0,3.0",
                    help="Target spacing as 'z,y,x' in mm (default 1.0,1.0,3.0)")
    ap.add_argument("--margin-mm", type=float, default=10.0,
                    help="Extra crop margin in mm around bbox (default 10mm)")
    ap.add_argument("--label-values", default=None,
                    help="Comma-separated integer labels to keep as foreground (default: >0)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only N rows (for dry runs)")
    return ap.parse_args()

def main():
    args = parse_args()
    spacing = tuple(float(s) for s in args.spacing.split(","))
    label_values = None
    if args.label_values:
        label_values = [int(s) for s in args.label_values.split(",")]

    from prostate_unet.preprocess import preprocess_pair  # local import after sys.path set by PyCharm Sources Root

    # Ensure all paths are correct - Check for absoluteness to avoid bugs due to incorrect pathing
    csv_path = Path(args.volumes_csv)
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_manifest = Path(args.out_manifest)
    if not out_manifest.is_absolute():
        out_manifest = (ROOT / out_manifest).resolve()

    kept = []
    for i, r in enumerate(rows):
        if args.limit and i >= args.limit:
            break
        sid = r.get("subject_week", f"row{i}")
        img_p = resolve_from_csv(r["image_path"], csv_path)
        lbl_p = resolve_from_csv(r["label_path"], csv_path)
        if not img_p.exists() or not lbl_p.exists():
            print(f"[skip] missing files for {sid}")
            continue

        img_nii = nib.load(str(img_p))
        lbl_nii = nib.load(str(lbl_p))

        out = preprocess_pair(
            img_nii=img_nii,
            lbl_nii=lbl_nii,
            target_spacing=spacing,
            label_values=label_values,
            margin_mm=args.margin_mm,
        )

        # cache path per subject
        cache_path = out_dir / f"{sid}.npz"
        np.savez_compressed(
            cache_path,
            image=out["image"],
            label=out["label"],
            spacing=out["spacing"],
            mean=out["mean"],
            std=out["std"],
        )

        kept.append({
            "subject_week": sid,
            "cache_path": str(cache_path),
            "shape_z": int(out["image"].shape[0]),
            "shape_y": int(out["image"].shape[1]),
            "shape_x": int(out["image"].shape[2]),
            "spacing_z": float(out["spacing"][0]),
            "spacing_y": float(out["spacing"][1]),
            "spacing_x": float(out["spacing"][2]),
        })
        print(f"[ok] {sid} -> {cache_path.name} {out['image'].shape} spacing={tuple(out['spacing'])}")

    # write manifest CSV
    with out_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0].keys()) if kept else
                           ["subject_week","cache_path","shape_z","shape_y","shape_x","spacing_z","spacing_y","spacing_x"])
        w.writeheader()
        for row in kept:
            w.writerow(row)

    print(f"[info] wrote {out_manifest} (rows={len(kept)})")

if __name__ == "__main__":
    main()