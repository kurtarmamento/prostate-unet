# Pair full volumes by shared prefix, e.g.:
# 'B006_Week0_LFOV.nii.gz'  <->  'B006_Week0_SEMANTIC.nii.gz'

import os, glob, re, csv, sys
DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
MR  = os.path.join(DATA_ROOT, "semantic_MRs")
LBL = os.path.join(DATA_ROOT, "semantic_labels_only")

def key(p: str):
    # Extract 'B006_Week0' from filenames like 'B006_Week0_LFOV.nii.gz'
    m = re.match(r"([A-Za-z0-9]+_Week\d+)_", os.path.basename(p))
    return m.group(1) if m else None

imgs = {key(p): p for p in glob.glob(os.path.join(MR, "*.nii.gz")) if key(p)}
lbls = {key(p): p for p in glob.glob(os.path.join(LBL, "*.nii.gz")) if key(p)}

keys = sorted(set(imgs) & set(lbls))

os.makedirs("artifacts", exist_ok=True)
out_csv = "artifacts/splits_volumes.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["subject_week","image_path","label_path"])
    for k in keys:
        w.writerow([k, imgs[k], lbls[k]])

print(f"Wrote {out_csv}  (pairs: {len(keys)})")