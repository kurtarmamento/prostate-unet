# NIfTI helpers: RAS Orientation and voxel sizes (zooms)

from __future__ import annotations
import nibabel as nib
import numpy as np
from typing import Tuple

def to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Return a new NIfTI reoriented to closest-RAS (canonical).
    Keeps data type; affine updated accordingly.
    """
    return nib.as_closest_canonical(img)

def voxel_sizes(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    """
    Voxel spacing in mm for the first 3 axes.
    """
    z = img.header.get_zooms()[:3]
    # Sometimes zooms can be numpy scalars; convert to float tuple
    return float(z[0]), float(z[1]), float(z[2])