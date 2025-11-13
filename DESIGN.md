# DESIGN

## Overview
This project trains a 2D/2.5D U-Net to segment the prostate on hip MRI.  
Key idea: treat each axial slice as a sample; for 2.5D, stack k neighboring slices as channels to add limited 3D context.

## Pipeline
1. **Index & Pair**  
   Pair MR and label volumes by subject/week; write absolute paths to `artifacts/splits_volumes.csv`.

2. **Preprocess** (`scripts/preprocess_volumes.py`)
   - Reorient to **RAS**
   - Resample to target spacing (default **1.0×1.0×3.0 mm**)
   - Compute bbox on binary label (+ **10 mm** margin); crop
   - Z-score intensities in the crop
   - Cache `.npz` with: `image`, `label`, `spacing`, `mean`, `std`, and **(for export)** `affine`, `orig_shape`, `crop_slices`

3. **Dataset** (`src/prostate_unet/datasets.py`)
   - 2D or 2.5D slices (k odd), center crop/pad to fixed size
   - Light augments (random flips), positive-slice sampling

4. **Model** (`src/prostate_unet/models/unet2d.py`)
   - U-Net with robust skip alignment (center-crop skips)
   - `in_ch = k` enables 2.5D

5. **Training** (`scripts/train_unet.py`)
   - Loss = **0.7·Dice + 0.3·BCE(stable)**
   - AdamW, grad-clip, ReduceLROnPlateau
   - Device picker: CUDA → DirectML → CPU; AMP on CUDA
   - Epoch-level ETA; best checkpoint saving

6. **Validation & Test** (`scripts/infer_volumes.py`)
   - Full-volume inference with robust paste-back
   - Threshold sweep (0.3–0.6)
   - Optional LCC
   - Writes per-volume CSV + split summary JSON

7. **Re-evaluation** (`scripts/eval_volumes.py`)
   - Re-score from saved probs without model forward

8. **Export** (`scripts/export_nifti.py`)
   - Paste crop back into full preprocessed grid using `orig_shape` and `crop_slices`
   - Save NIfTI with cached `affine`

## Design Decisions (high-level)
- **2.5D vs 3D**: 2.5D is simpler and GPU-friendly, good baseline before moving to 3D.
- **Fixed crop size**: enables stable batching; center crop/pad handles edge cases.
- **Stable BCE**: avoids DirectML `log_sigmoid` fallbacks; DML runs mostly on GPU.
- **Threshold via validation**: prevents test-set tuning; improves generalization.
- **LCC post-proc**: simple heuristic to reduce spurious blobs.

## Key Parameters (baseline)
- Spacing: **1.0×1.0×3.0 mm**
- Crop size: **256×256**
- k-slices: **5** (2.5D)
- Base channels: **32**
- LR: **1e-3**, AdamW, grad-clip 1.0
- Margin: **10 mm**
- **Chosen global threshold (from val): 0.30**

## Known Limitations
- 2.5D may miss long-range 3D context.
- No domain adaptation / intensity harmonization across scanners.
- Simple augmentations only.

## Next Steps (v2)
- True **3D U-Net** with patch-based training + sliding-window inference
- Stronger augmentation (elastic, gamma), mixup/cutmix (2D)
- Calibrated thresholding or CRF-style refinement
