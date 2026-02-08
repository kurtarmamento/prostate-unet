# tests/test_smoke.py
from __future__ import annotations

from pathlib import Path
import sys
import torch
import torch.nn as nn

# Make `prostate_unet` importable without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prostate_unet.models.unet2d import UNet2D  # noqa: E402


def test_unet2d_forward_shape():
    model = UNet2D(in_ch=1, base=16)
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)


def test_unet2d_backward_step_is_finite():
    torch.manual_seed(0)
    model = UNet2D(in_ch=1, base=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    x = torch.randn(2, 1, 128, 128)
    y = (torch.rand(2, 1, 128, 128) > 0.8).float()

    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)

    assert torch.isfinite(loss).item()
    loss.backward()
    opt.step()
