# Preprocessing primitives: resample to target spacing, crop to bbox, cache.

from __future__ import annotations
from typing import Tuple, Sequence, Optional, Dict
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
from .io_nifti import to_ras, voxel_sizes


def resample_to_spacing(
        vol: np.ndarray,
        current_spacing: Sequence[float],
        target_spacing: Sequence[float],
        order: int,
) -> np.ndarray:
    """
    Resample a 3D volume to the target voxel spacing using scipy.ndimage.zoom.

    :param vol: (Z, Y, X) ndarray
    :param current_spacing: (sz, sy, sx) in mm (spacing of 'vol')
    :param target_spacing: (tz, ty, tx) in mm (desired spacing)
    :param order: 1 for images (linear), 0 for labels (nearest)
    :return: Resampled volume as ndarray (still (Z, Y, X))
    """
    cs = np.array(current_spacing, dtype=np.float64)
    ts = np.array(target_spacing, dtype=np.float64)
    # zoom factor = old_spacing / new_spacing (e.g. 2.0mm -> 1.0mm => 2.0x)
    factors = cs / ts
    return zoom(vol, zoom=factors, order=order, mode="nearest")


def bbox_from_mask(mask: np.ndarray, margin_vox: Sequence[int]) -> Tuple[slice, slice, slice]:
    """
    Compute a tight bounding box around mask > 0 with an extra margin (in voxels).
    Returns a tuple of slices usable for vol[z, y, x] indexing.

    :param mask: (Z, Y, X) ndarray of 0/1
    :param margin_vox: (mz, my, mx) margin (voxels) to extend bbox
    :return: tuple of slices (z_slice, y_slice, x_slice)
    """
    assert mask.ndim == 3
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        # no foreground: return full volume
        return slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2])

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    mz, my, mx = margin_vox

    z0 = max(0, zmin - mz); z1 = min(mask.shape[0], zmax + mz + 1)
    y0 = max(0, ymin - my); y1 = min(mask.shape[1], ymax + my + 1)
    x0 = max(0, xmin - mx); x1 = min(mask.shape[2], xmax + mx + 1)
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def _affine_from_target_spacing(target_spacing: Sequence[float]) -> np.ndarray:
    """
    Build a simple affine for the resampled grid.

    NOTE: Arrays here are (Z, Y, X). NIfTI affine columns correspond to X,Y,Z.
    We put target spacings on the diagonal in (X,Y,Z) order -> diag(tx, ty, tz, 1).
    This yields a consistent metric grid for exporting masks in the preprocessed space.
    """
    tz, ty, tx = map(float, target_spacing)  # incoming order is (z,y,x)
    aff = np.eye(4, dtype=np.float32)
    aff[0, 0] = tx
    aff[1, 1] = ty
    aff[2, 2] = tz
    # zero translation; sufficient for preprocessed-space exports
    return aff


def preprocess_pair(
    img_nii: nib.Nifti1Image,
    lbl_nii: nib.Nifti1Image,
    target_spacing=(1.0, 1.0, 3.0),
    label_values: Optional[Sequence[int]] = None,
    margin_mm: float = 10.0,
) -> Dict[str, np.ndarray]:
    """
    Full pipeline for one (image,label) pair:
      1) orient to RAS
      2) resample image (order=1) and label (order=0) to target_spacing
      3) crop to bbox(mask>0) + margin
      4) z-score intensity inside the crop
      5) return arrays + metadata (incl. affine/orig_shape/crop_slices) ready to cache
    """
    # 1) orient to RAS (consistent axis order/orientation)
    img_ras = to_ras(img_nii)
    lbl_ras = to_ras(lbl_nii)

    # 2) build binary label mask (>0 or specific classes)
    cur_sz = voxel_sizes(img_ras)  # (sz, sy, sx)
    lbl_data = lbl_ras.get_fdata()
    if label_values is not None:
        mask_full = np.isin(lbl_data, list(label_values))
    else:
        mask_full = lbl_data > 0

    # resample image and mask to target spacing
    img_res = resample_to_spacing(img_ras.get_fdata().astype(np.float32), cur_sz, target_spacing, order=1)
    msk_res = resample_to_spacing(mask_full.astype(np.uint8),           cur_sz, target_spacing, order=0)

    # record pre-crop shape and construct an affine for the resampled grid
    orig_shape = np.array(img_res.shape, dtype=np.int32)        # (Z, Y, X) before crop
    aff_res    = _affine_from_target_spacing(target_spacing)    # 4x4 (pre-crop)

    # 3) crop to bbox + margin (convert mm -> vox based on target spacing)
    margin_vox = tuple(int(round(margin_mm / s)) for s in target_spacing)
    zsl, ysl, xsl = bbox_from_mask(msk_res, margin_vox)
    img_crop = img_res[zsl, ysl, xsl]
    msk_crop = msk_res[zsl, ysl, xsl].astype(np.uint8)

    # save explicit crop indices for paste-back
    z0, z1 = zsl.start, zsl.stop
    y0, y1 = ysl.start, ysl.stop
    x0, x1 = xsl.start, xsl.stop
    crop_slices = np.array([z0, z1, y0, y1, x0, x1], dtype=np.int32)

    # 4) z-score normalization inside the crop (optional but standard)
    mu = float(img_crop.mean()); sigma = float(img_crop.std() + 1e-6)
    img_norm = (img_crop - mu) / sigma

    # 5) return payload for caching
    return {
        "image":    img_norm.astype(np.float32),                 # (Zc, Yc, Xc)
        "label":    msk_crop,                                    # 0/1 after selection
        "spacing":  np.array(target_spacing, dtype=np.float32),  # (tz, ty, tx)
        "mean":     mu,
        "std":      sigma,
        # Day-4 metadata for export/inference tooling:
        "affine":      aff_res.astype(np.float32),               # 4x4 affine for resampled pre-crop grid
        "orig_shape":  orig_shape,                               # (Z, Y, X) before crop
        "crop_slices": crop_slices,                              # [z0,z1,y0,y1,x0,x1]
    }
