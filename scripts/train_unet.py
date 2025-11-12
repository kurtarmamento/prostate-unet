# scripts/train_unet.py
from pathlib import Path
import sys, csv, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- import src/ ---
ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prostate_unet.datasets import Slices2D
from prostate_unet.models.unet2d import UNet2D

def pick_device():
    """
    Prefer CUDA (incl. ROCm builds on Linux), else DirectML on Windows (AMD),
    else CPU. AMP only on CUDA.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", True  # use_amp
    try:
        import torch_directml  # noqa: F401
        dml = torch_directml.device()  # DirectML device
        return dml, "dml", False       # AMP off on DML
    except Exception:
        pass
    return torch.device("cpu"), "cpu", False

def bce_with_logits_stable(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    DML-friendly BCE with logits: max(x,0) - x*y + log1p(exp(-|x|))
    targets in {0,1}, logits float.
    """
    x = logits
    y = targets
    # same as F.binary_cross_entropy_with_logits but without log_sigmoid
    neg_abs = -x.abs()
    loss = torch.clamp(x, min=0) - x * y + torch.log1p(torch.exp(neg_abs))
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss using probabilities = sigmoid(logits).
    logits: (N,1,H,W), targets: (N,1,H,W) in {0,1}
    """
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1,2,3))
    denom = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (denom + eps)
    return 1.0 - dice.mean()



def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2.0 * inter + eps) / (denom + eps)).mean().item()

@torch.no_grad()
def quick_slice_dice(model, val_loader, device, use_amp: bool, max_batches: int = 3) -> float:
    """
    Averaged slice-level Dice over a few val batches in eval() mode.
    Reduces noise vs. using only the first batch.
    """
    import torch
    was_training = model.training
    model.eval()
    try:
        tot, n = 0.0, 0
        for i, (x, y, _) in enumerate(val_loader):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    p = (torch.sigmoid(model(x)) >= 0.5).float()
            else:
                p = (torch.sigmoid(model(x)) >= 0.5).float()
            tot += dice_coeff(p, y); n += 1
        return tot / max(1, n)
    finally:
        if was_training:
            model.train()


def train_one_epoch(model, loader, opt, device, use_amp: bool, epoch:int, epochs:int) -> float:
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    n_batches = len(loader)
    t0 = time.time()
    ema_bt = None
    loss_running = 0.0

    for bi, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                # DML-friendly losses:
                loss_bce  = bce_with_logits_stable(logits, y, reduction="mean")
                loss_dice = dice_loss_from_logits(logits, y)
                loss = 0.5 * loss_bce + 0.5 * loss_dice
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss_bce  = bce_with_logits_stable(logits, y, reduction="mean")
            loss_dice = dice_loss_from_logits(logits, y)
            loss = 0.3 * loss_bce + 0.7 * loss_dice
            loss.backward()
            opt.step()

        loss_running += loss.item() * x.size(0)

        # --- ETA display (update every ~10% of epoch) ---
        bt = time.time() - t0 if bi == 1 else (time.time() - last_tick)
        ema_bt = bt if ema_bt is None else 0.9 * ema_bt + 0.1 * bt
        last_tick = time.time()
        if bi == 1 or bi % max(1, n_batches // 10) == 0 or bi == n_batches:
            left_batches = (n_batches - bi) + (epochs - epoch) * n_batches
            eta_s = left_batches * (ema_bt if ema_bt is not None else bt)
            mins, secs = divmod(int(eta_s), 60)
            print(f"  [epoch {epoch}/{epochs}] batch {bi}/{n_batches} "
                  f"| loss {loss.item():.4f} | ETA ~ {mins:02d}m{secs:02d}s")

    return loss_running / len(loader.dataset)

@torch.no_grad()
def validate_volumes(model, manifest_csv: str | Path, device,
                     crop_size: int = 256, k: int = 1, split: str = "val",
                     use_amp: bool = False, debug: bool = False, debug_n: int = 2):
    """
    Per-volume validation in true eval() mode:
      - robustly pastes predictions back into original (h,w) for all crop/pad cases
      - scans thresholds (0.3, 0.4, 0.5, 0.6) and keeps the best
      - optional largest connected component denoising
      - tiny debug print for volumes that collapse to ~0 Dice
    Returns: (mean_best_dice_across_volumes, num_volumes)
    """
    import numpy as np, csv
    from pathlib import Path
    import torch

    # ensure eval() (don’t update BatchNorm running stats)
    was_training = model.training
    model.eval()

    # rows for the requested split
    rows = list(csv.DictReader(Path(manifest_csv).open()))
    rows = [r for r in rows if r.get("split", "val") == split]
    if not rows:
        if was_training: model.train()
        raise ValueError(f"No rows for split={split} in {manifest_csv}")

    # optional LCC (safe if SciPy missing)
    try:
        from scipy.ndimage import label as lcc_label
        def keep_lcc(bin3d: np.ndarray) -> np.ndarray:
            lab, n = lcc_label(bin3d)
            if n == 0: return bin3d
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            return (lab == sizes.argmax()).astype(np.uint8)
    except Exception:
        def keep_lcc(bin3d: np.ndarray) -> np.ndarray:
            return bin3d

    dices = []
    zeros_reported = 0

    try:
        for r in rows:
            d = np.load(r["cache_path"])
            vol = d["image"].astype(np.float32)  # (Z, h, w)
            msk = d["label"].astype(np.uint8)    # (Z, h, w)
            Z, h, w = vol.shape
            H = W = crop_size

            preds = []
            for z in range(Z):
                # ---- build k-slice context ----
                if k == 1:
                    sl = vol[z:z+1]  # (1,h,w)
                else:
                    half = k // 2
                    zs = [min(max(zz, 0), Z - 1) for zz in range(z - half, z + half + 1)]
                    sl = vol[zs]     # (k,h,w)

                # ---- center-crop if larger ----
                y0 = max(0, (h - H) // 2)
                x0 = max(0, (w - W) // 2)
                sl_crop = sl[:, y0:y0+H, x0:x0+W]

                # ---- pad if smaller (mirror train loader) ----
                if sl_crop.shape[1] < H or sl_crop.shape[2] < W:
                    C, hh, ww = sl_crop.shape
                    ph = max(0, H - hh); pw = max(0, W - ww)
                    top = ph // 2; bottom = ph - top
                    left = pw // 2; right = pw - left
                    sl_crop = np.pad(sl_crop, ((0,0),(top,bottom),(left,right)), mode="edge")

                # ---- run model ----
                x = torch.from_numpy(sl_crop[None, ...]).to(device, non_blocking=True)  # (1,C,H,W)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        p = torch.sigmoid(model(x)).float().cpu().numpy()[0, 0]  # (H,W)
                else:
                    p = torch.sigmoid(model(x)).float().cpu().numpy()[0, 0]      # (H,W)

                # ---- robust paste-back into (h,w) ----
                pad_y = max(0, (H - h) // 2)
                pad_x = max(0, (W - w) // 2)

                if h >= H:
                    y_src0, y_src1 = 0, H
                    y_dst0, y_dst1 = y0, y0 + H
                else:
                    y_src0, y_src1 = pad_y, pad_y + h
                    y_dst0, y_dst1 = 0, h

                if w >= W:
                    x_src0, x_src1 = 0, W
                    x_dst0, x_dst1 = x0, x0 + W
                else:
                    x_src0, x_src1 = pad_x, pad_x + w
                    x_dst0, x_dst1 = 0, w

                canvas = np.zeros((h, w), dtype=np.float32)
                canvas[y_dst0:y_dst1, x_dst0:x_dst1] = p[y_src0:y_src1, x_src0:x_src1]
                preds.append(canvas)

            pred3d = np.stack(preds, axis=0)  # (Z,h,w)

            # ---- threshold sweep (+ LCC) -> best Dice for this volume ----
            best_d = 0.0
            for t in (0.3, 0.4, 0.5, 0.6):
                bin3d = (pred3d >= t).astype(np.uint8)
                bin3d = keep_lcc(bin3d)
                inter = (bin3d & msk).sum()
                denom = bin3d.sum() + msk.sum()
                best_d = max(best_d, (2.0 * inter) / (denom + 1e-6))

            dices.append(best_d)

            # minimal debug if a volume collapsed
            if best_d <= 0.01 and debug and zeros_reported < debug_n:
                nz_pred = int((pred3d >= 0.5).sum() > 0)
                nz_msk  = int(msk.sum() > 0)
                print(f"[val-debug] {r.get('subject_week','?')}: best={best_d:.3f}, "
                      f"pred>0={nz_pred}, msk>0={nz_msk}, shape={pred3d.shape}")
                zeros_reported += 1

        return float(np.mean(dices)), len(dices)
    finally:
        if was_training:
            model.train()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-csv", default=str(ROOT / "artifacts" / "cache_splits.csv"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--k-slices", type=int, default=1)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device, devtype, use_amp = pick_device()
    print(f"[device] {devtype} | AMP={'on' if use_amp else 'off'}")

    if devtype == "cuda":
        torch.backends.cudnn.benchmark = True

    model = UNet2D(in_ch=args.k_slices, base=args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    pin = (devtype in ("cuda", "dml"))
    train_ds = Slices2D(args.splits_csv, split="train", crop_size=args.crop_size, k_slices=args.k_slices)
    val_ds   = Slices2D(args.splits_csv, split="val",   crop_size=args.crop_size, k_slices=args.k_slices)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    best = -1.0
    ckpt = ROOT / "artifacts" / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_ld, opt, device, use_amp, epoch, args.epochs)

        sl_dice = quick_slice_dice(model, val_ld, device, use_amp, max_batches=3)

        val_dice, nvol = validate_volumes(
            model, args.splits_csv, device,
            crop_size=args.crop_size, k=args.k_slices, split="val", use_amp=use_amp
        )

        print(f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} | "
              f"val_slice_dice {sl_dice:.3f} | val_vol_dice {val_dice:.3f} ({nvol} vols)")

        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), ckpt / "unet2d_best.pth")
            print(f"  -> saved checkpoint (best val_vol_dice={best:.3f})")

    print("Done. Best val vol Dice:", best)

if __name__ == "__main__":
    main()
