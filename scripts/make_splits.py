from pathlib import Path
import csv, hashlib

ROOT = Path(__file__).resolve().parent.parent
in_csv  = ROOT/"artifacts"/"cache_volumes.csv"
out_csv = ROOT/"artifacts"/"cache_splits.csv"

def subj_of(sid:str)->str:
    # 'K018_Week0' -> 'K018'
    return sid.split("_Week")[0]

rows = list(csv.DictReader(in_csv.open()))
by_subj = {}
for r in rows:
    s = subj_of(r["subject_week"])
    by_subj.setdefault(s, []).append(r)

def bucket(subject:str)->str:
    # stable 70/15/15 split via hash
    h = int(hashlib.md5(subject.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"

out_fields = list(rows[0].keys()) + ["split"]
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_fields); w.writeheader()
    for s, items in by_subj.items():
        sp = bucket(s)
        for r in items:
            rr = dict(r); rr["split"] = sp
            w.writerow(rr)

print(f"Wrote {out_csv} with {len(rows)} rows.")
