# scripts/preprocess_volumes.py
# Iterate artifacts/splits_volumes.csv, preprocess each pair, and cache .npz files + a manifest.
# Notes:
# - Preserves your original flow (args, path resolution, prints, manifest).
# - Saves extra Day-4 metadata (affine, orig_shape, crop_slices) if provided by preprocess_pair(...).
#   These are needed for exporting full-size NIfTI masks later.

from __future__ import annotations

import sys
from pathlib import Path
import argparse, csv
import numpy as np
import nibabel as nib

# --- Project roots / paths (same as yours) ---
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


# --- Utilities ---
def resolve_from_csv(pth: str, csv_path: Path) -> Path:
    """
    Resolve a path possibly stored relative to the CSV's folder.
    Keeps absolute paths intact; resolves relatives against the CSV location.
    """
    p = Path(pth.strip())
    return p if p.is_absolute() else (csv_path.parent / p).resolve()


def parse_args():
    """
    CLI options (unchanged where possible):
      --volumes-csv   : CSV with columns image_path,label_path,subject_week
      --out-dir       : where to write cached .npz files
      --out-manifest  : CSV manifest of cached files
      --spacing       : target voxel spacing 'z,y,x' (mm)
      --margin-mm     : bbox margin around prostate (mm)
      --label-values  : e.g. '5' to isolate prostate=5 and binarize
      --limit         : process only the first N rows
    """
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

    # Parse spacing "z,y,x" -> tuple[float, float, float]
    spacing = tuple(float(s) for s in args.spacing.split(","))
    if len(spacing) != 3:
        raise ValueError(f"--spacing must be 'z,y,x' (got {args.spacing})")

    # Parse label filter if provided (e.g., "5" to keep label==5 only)
    label_values = None
    if args.label_values:
        label_values = [int(s) for s in args.label_values.split(",")]

    # Local import after sys.path tweak so PyCharm/Windows finds it
    from prostate_unet.preprocess import preprocess_pair  # must return image/label/spacing/mean/std; optionally affine/orig_shape/crop_slices

    # ---- Resolve input CSV path and read rows ----
    csv_path = Path(args.volumes_csv)
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # ---- Prepare output destinations ----
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

        # Subject id (falls back to row index if field missing)
        sid = r.get("subject_week", f"row{i}")

        # Resolve image/label paths (absolute or relative to CSV)
        img_p = resolve_from_csv(r["image_path"], csv_path)
        lbl_p = resolve_from_csv(r["label_path"], csv_path)
        if not img_p.exists() or not lbl_p.exists():
            print(f"[skip] {sid}: missing files (img={img_p.exists()} lbl={lbl_p.exists()})")
            continue

        # Load NIfTI headers/data (lazy until get_fdata() inside preprocess_pair)
        try:
            img_nii = nib.load(str(img_p))
            lbl_nii = nib.load(str(lbl_p))
        except Exception as e:
            print(f"[skip] {sid}: nibabel load error -> {e}")
            continue

        # ---- Core preprocessing (reorient/resample/bbox-crop/normalize) ----
        # NOTE: For Day-4 export, preprocess_pair should also return:
        #   'affine' (4x4, pre-crop), 'orig_shape' (Z,H,W pre-crop), 'crop_slices' [z0,z1,y0,y1,x0,x1]
        out = preprocess_pair(
            img_nii=img_nii,
            lbl_nii=lbl_nii,
            target_spacing=spacing,
            label_values=label_values,
            margin_mm=args.margin_mm,
        )

        # ---- Cache to .npz (always save the essentials) ----
        cache_path = out_dir / f"{sid}.npz"

        # Prepare base payload (what your original script saved)
        payload = {
            "image": out["image"].astype(np.float32),                # cropped, z-scored image (Zc,Hc,Wc)
            "label": out["label"].astype(np.uint8),                  # cropped mask (0/1)
            "spacing": np.asarray(out["spacing"], dtype=np.float32), # voxel spacing (z,y,x) mm
            "mean": float(out["mean"]),
            "std": float(out["std"]),
        }

        # Optional Day-4 metadata (saved only if provided by preprocess_pair)
        missing_keys = []
        for k, dtype in (("affine", np.float32), ("orig_shape", np.int32), ("crop_slices", np.int32)):
            if k in out:
                payload[k] = np.asarray(out[k], dtype=dtype)
            else:
                missing_keys.append(k)

        if missing_keys:
            # Non-fatal: you can still train/infer; only NIfTI export requires these.
            print(f"[warn] {sid}: preprocess_pair did not return {missing_keys} "
                  f"(export_nifti.py will be unavailable for this case)")

        np.savez_compressed(cache_path, **payload)

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

    # ---- Write manifest CSV (same columns you already used) ----
    with out_manifest.open("w", newline="") as f:
        fieldnames = list(kept[0].keys()) if kept else \
            ["subject_week","cache_path","shape_z","shape_y","shape_x","spacing_z","spacing_y","spacing_x"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in kept:
            w.writerow(row)

    print(f"[info] wrote {out_manifest} (rows={len(kept)})")


if __name__ == "__main__":
    main()
