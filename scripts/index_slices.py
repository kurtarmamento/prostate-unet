# Build a CSV manifest of paired SLICE image/mask files

import os, glob, re, csv, sys

# Get directory of data from arguments; combine paths for data specifically
DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

ROOT = os.path.join(DATA_ROOT, "keras_slices_data")

def grab(split: str):
    """
    For a given split (train/validate/test), find image and label files, then pair them by a shared key
    :param split: String -> train; validate; test
    """

    # Find all files following the naming convention in ROOT Path
    imgs = glob.glob(os.path.join(ROOT, f"keras_slices_{split}", "case_*.nii.gz"))
    lbls = glob.glob(os.path.join(ROOT, f"keras_slices_seg_{split}", "seg_*.nii.gz"))

    # 'case_004_week_0_slice_2.nii.gz' -> '004_week_0_slice_2'
    m_img = {
        re.sub(r"^case_", "", os.path.basename(p)).replace(".nii.gz", ""): os.path.abspath(p)
        for p in imgs
    }
    # 'seg_004_week_0_slice_2.nii.gz'  -> '004_week_0_slice_2'
    m_lbl = {
        re.sub(r"^seg_", "", os.path.basename(p)).replace(".nii.gz", ""): os.path.abspath(p)
        for p in lbls
    }

    # Intersect keys to keep only well-formed pairs
    for k in sorted(set(m_img) & set(m_lbl)):
        # Optional readability: extract parts from the key
        m = re.match(r"(\d+)_week_(\d+)_slice_(\d+)$", k)
        case, week, sl = m.groups() if m else ("", "", "")
        yield [split, case, week, sl, m_img[k], m_lbl[k]]

# Write one CSV that lists every paired slice across all three splits
os.makedirs("artifacts", exist_ok=True)
out_csv = "artifacts/splits_slices.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split","case_id","week","slice_idx","image_path","label_path"])
    for sp in ["train","validate","test"]:
        for row in grab(sp):
            w.writerow(row)

print(f"Wrote {out_csv}")