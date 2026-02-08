# scripts/smoke_train_dummy_step.py
"""
Smoke training: run a few optimization steps on synthetic tensors.

This validates:
- model forward pass
- loss computation
- backward pass
- optimizer step

No dataset required.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prostate_unet.models.unet2d import UNet2D  # noqa: E402


def resolve_device(device: str) -> torch.device:
    d = (device or "auto").lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if d == "directml":
        if sys.platform != "win32":
            raise RuntimeError("DirectML is only supported on Windows.")
        try:
            import torch_directml  # type: ignore
        except ImportError as e:
            raise RuntimeError("DirectML requested but torch-directml is not installed.") from e
        return torch_directml.device()
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke training on synthetic tensors.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "directml"])
    ap.add_argument("--k-slices", type=int, default=1)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.k_slices % 2 != 1:
        raise SystemExit("--k-slices must be odd (1,3,5,...)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)

    model = UNet2D(in_ch=args.k_slices, base=args.base_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # synthetic batch
    B = 2
    x = torch.randn(B, args.k_slices, args.crop_size, args.crop_size, device=device)
    # synthetic circular-ish mask in the center
    yy, xx = torch.meshgrid(
        torch.arange(args.crop_size, device=device),
        torch.arange(args.crop_size, device=device),
        indexing="ij",
    )
    cy = args.crop_size // 2
    cx = args.crop_size // 2
    r = args.crop_size // 6
    mask = (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).float()
    y = mask.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,1,H,W)

    model.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        opt.step()
        print(f"step {step}/{args.steps} loss={loss.item():.6f}")

    print("[ok] smoke training completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
