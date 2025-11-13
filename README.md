# prostate-unet (2.5D baseline)

Baseline 2D/2.5D U-Net for prostate segmentation on hip MRI volumes.  
**Not for clinical use.** Research/education only.

## TL;DR
- Preprocess: RAS → resample to target spacing → bbox crop → cache `.npz`
- Train: 2D or 2.5D (k-slice stacks), Dice + BCE (stable), DirectML/CUDA/CPU
- Evaluate: full-volume inference, threshold sweep, optional LCC
- Export: predicted masks to full-size NIfTI for inspection

## Data layout (expected)
```
semantic_MRs/            # NIfTI MR volumes (*.nii.gz)
semantic_labels_only/    # matching label volumes (*.nii.gz)
artifacts/               # generated: cache, preds, previews, checkpoints, etc.
```
> If your prostate is a specific label (e.g. **5**), pass `--label-values 5` during preprocessing.

## Quick start (Windows/PowerShell shown)

### 1) Preprocess volumes (cache cropped volumes)
```powershell
python scripts\preprocess_volumes.py ^
  --volumes-csv artifacts\splits_volumes.csv ^
  --spacing 1.0,1.0,3.0 ^
  --margin-mm 10 ^
  --label-values 5
```

### 2) Train (example 2.5D config)
```powershell
python scripts\train_unet.py ^
  --epochs 800 --batch-size 12 --crop-size 256 ^
  --k-slices 5 --base-ch 32 --lr 1e-3
```

### 3) Infer on validation (find best threshold)
```powershell
python scripts\infer_volumes.py ^
  --split val ^
  --ckpt artifacts\checkpoints\unet2d_best.pth ^
  --k-slices 5 --crop-size 256 --lcc ^
  --outdir artifacts\preds_val
```
Read `artifacts/preds_val/summary_val.json` → **best_global_threshold**.

### 4) Infer on test (use the val threshold = 0.30)
```powershell
python scripts\infer_volumes.py ^
  --split test ^
  --ckpt artifacts\checkpoints\unet2d_best.pth ^
  --k-slices 5 --crop-size 256 --lcc ^
  --threshold 0.3 ^
  --outdir artifacts\preds_test
```

### 5) (Optional) Re-score without re-running the model
```powershell
python scripts\eval_volumes.py ^
  --split test ^
  --preds_dir artifacts\preds_test ^
  --threshold 0.3 ^
  --lcc
```

### 6) (Optional) Export full-size NIfTI masks
```powershell
python scripts\export_nifti.py ^
  --split test ^
  --preds_dir artifacts\preds_test ^
  --threshold 0.3 ^
  --lcc ^
  --outdir artifacts\nifti_out
```

## Results
```
| Split | Threshold |  LCC | Mean Dice |
|------:|:---------:|:----:|----------:|
|  Val  |   0.30    |  ✅  |   0.79    |
|  Test |   0.30    |  ✅  |   0.52    |
```

## Repo structure
```
src/prostate_unet/
  models/unet2d.py
  datasets.py
  preprocess.py
  utils/...
scripts/
  preprocess_volumes.py
  train_unet.py
  infer_volumes.py
  eval_volumes.py
  export_nifti.py
artifacts/  # generated (cache, preds_*, previews, checkpoints, etc.)
```

## Requirements
- Python 3.11
- PyTorch (CUDA or CPU). AMD on Windows supported via DirectML.
- nibabel, numpy, scipy, tqdm (see `pyproject.toml` / `requirements.txt`)

## Disclaimer
This repository is for research/education only and **must not** be used for diagnosis or patient care.

## Roadmap
- v1 baseline (this): 2.5D U-Net, full-volume eval/export
- v1.1: small ablations (crop size, base channels, LCC sensitivity)
- v2: 3D U-Net + sliding-window inference
