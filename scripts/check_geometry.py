# Verify image/mask shape + affine equality for a few pairs from the CSVs.
# Works regardless of current working directory and regardless of absolute/relative paths in the CSV.

from pathlib import Path
import csv
import nibabel as nib
import numpy as np

# --- locate artifacts/ robustly (scripts/ and artifacts/ are siblings) ---
HERE = Path(__file__).resolve().parent       # .../prostate-unet/scripts
ROOT = HERE.parent                           # .../prostate-unet
ARTIFACTS = ROOT / "artifacts"
VOL_CSV = ARTIFACTS / "splits_volumes.csv"
SLICE_CSV = ARTIFACTS / "splits_slices.csv"

def resolve_from_csv(pth: str, csv_path: Path) -> Path:
    """
    Return an absolute Path for a CSV field value.
    If the path in the CSV is relative, resolve it relative to the CSV's folder.
    """
    p = Path(pth.strip())
    return p if p.is_absolute() else (csv_path.parent / p).resolve()

def check_csv(csv_path: Path, n: int = 5, affine_atol: float = 1e-3, label_cols=("label_path",)):
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        return

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"[info] loaded {len(rows)} rows from {csv_path.name}")

    # detect column names (volumes vs slices)
    has_subject = "subject_week" in rows[0] if rows else False
    id_fmt = (lambda r: r.get("subject_week", "?")) if has_subject else \
             (lambda r: f"{r.get('split','?')} case={r.get('case_id','?')} slice={r.get('slice_idx','?')}")

    for r in rows[:n]:
        img_p = resolve_from_csv(r["image_path"], csv_path)
        # support either "label_path" or "lbl_path" if you changed headers later
        lbl_key = "label_path" if "label_path" in r else ("lbl_path" if "lbl_path" in r else "label")
        lbl_p = resolve_from_csv(r[lbl_key], csv_path)

        if not img_p.exists() or not lbl_p.exists():
            print(f"[miss] cannot find file(s):\n  img={img_p} (exists={img_p.exists()})\n  lbl={lbl_p} (exists={lbl_p.exists()})")
            continue

        img = nib.load(str(img_p))
        lbl = nib.load(str(lbl_p))

        same_shape = (img.shape == lbl.shape)
        same_aff   = np.allclose(img.affine, lbl.affine, atol=affine_atol)
        vox_img = img.header.get_zooms()[:3]
        vox_lbl = lbl.header.get_zooms()[:3]
        print(f"ok={same_shape and same_aff} shape={img.shape} vox_img={vox_img} vox_lbl={vox_lbl} :: {id_fmt(r)}")

print("Volumes:")
check_csv(VOL_CSV, n=5)

print("\nSlices:")
check_csv(SLICE_CSV, n=5)
