# src/prostate_unet/models/unet2d.py
import torch
import torch.nn as nn

def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

def _center_crop_to(t: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """
    Center-crop tensor t to spatial size target_hw=(H,W). Assumes t is (N,C,H,W)
    and t is at least as large as target in both dims (typical in U-Net skips).
    """
    _, _, H, W = t.shape
    th, tw = target_hw
    dh, dw = H - th, W - tw
    if dh == 0 and dw == 0:
        return t
    top = max(0, dh // 2); left = max(0, dw // 2)
    bottom = H - (dh - top); right = W - (dw - left)
    return t[:, :, top:bottom, left:right]

class UNet2D(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        # Encoder
        self.down1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        # Bottleneck
        self.bridge = conv_block(base * 4, base * 8)
        # Decoder
        self.up3  = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1  = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.out  = nn.Conv2d(base, 1, kernel_size=1)  # logits (N,1,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b  = self.bridge(self.pool3(d3))

        u3 = self.up3(b)
        if d3.shape[-2:] != u3.shape[-2:]:
            d3 = _center_crop_to(d3, u3.shape[-2:])
        u3 = self.dec3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(u3)
        if d2.shape[-2:] != u2.shape[-2:]:
            d2 = _center_crop_to(d2, u2.shape[-2:])
        u2 = self.dec2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        if d1.shape[-2:] != u1.shape[-2:]:
            d1 = _center_crop_to(d1, u1.shape[-2:])
        u1 = self.dec1(torch.cat([u1, d1], dim=1))

        return self.out(u1)
