from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
cache_dir = ROOT / "artifacts" / "cache"
sample = next(cache_dir.glob("*.npz"))
arr = np.load(sample)
img, msk = arr["image"], arr["label"]
z = img.shape[0] // 2

plt.figure(figsize=(4,4)); plt.axis("off")
sl = (img[z] - img[z].mean()) / (img[z].std() + 1e-6)
plt.imshow(sl, cmap="gray")
plt.contour(msk[z].astype(float), levels=[0.5], colors="red", linewidths=1)
plt.tight_layout(pad=0); plt.show()
print("sample:", sample.name, "shape:", img.shape, "spacing:", arr["spacing"])
