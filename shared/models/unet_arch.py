"""
shared.models.unet_arch — PyTorch U-Net architecture for radio map completion.

Used ONLY during training and ONNX export.
At runtime, master uses the exported ONNX via OnnxBackend — no PyTorch required.

Input:  [B, 5, H, W]
  ch 0: DEM (normalised 0-1)
  ch 1: building height (normalised 0-1, zeros if no OSM data)
  ch 2: vegetation attenuation (0-1, zeros if no Corine data)
  ch 3: sparse RSSI map (0 where unmeasured, normalised RSSI elsewhere)
  ch 4: measurement mask (1 where an RSSI observation exists, else 0)

Output: [B, 1, H, W] — dense radio map, normalised to [0, 1]

Requirements:
  H and W must be divisible by 16 (4 pooling steps).
  Recommended: H = W = 64 for training, 256 for fine-tuning.

Architecture: standard U-Net with 4 encoder/decoder levels.
  base=32  → ~3.1M parameters
  base=16  → ~800k parameters (faster training, lower quality)
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for UNet training/export. "
            "Install with: pip install torch"
        )


class _DoubleConv(nn.Module):
    """Two (Conv → BN → ReLU) blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class UNetRadioMap(nn.Module):
    """
    U-Net for radio map completion from sparse RSSI + terrain conditioning.

    Input:  [B, in_ch, H, W]  (default in_ch=5)
    Output: [B, 1,     H, W]  in [0, 1]

    dropout: spatial dropout applied after the bottleneck (0 = disabled).
             Use 0.1–0.3 to regularize against over-fitting noisy sim data.
    """

    def __init__(self, in_ch: int = 5, base: int = 32, dropout: float = 0.0) -> None:
        _require_torch()
        super().__init__()

        # Encoder
        self.enc1 = _DoubleConv(in_ch, base)
        self.enc2 = _DoubleConv(base, base * 2)
        self.enc3 = _DoubleConv(base * 2, base * 4)
        self.enc4 = _DoubleConv(base * 4, base * 8)

        # Bottleneck
        self.bottleneck = _DoubleConv(base * 8, base * 16)
        self.bottleneck_drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(base, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck_drop(self.bottleneck(self.pool(e4)))

        # Decoder path with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


def build_unet(base: int = 32, in_ch: int = 5, dropout: float = 0.0) -> "UNetRadioMap":
    """Factory: create a UNetRadioMap and return it."""
    _require_torch()
    return UNetRadioMap(in_ch=in_ch, base=base, dropout=dropout)


def count_parameters(model: "nn.Module") -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_to_onnx(
    model: "UNetRadioMap",
    output_path: str,
    grid_size: int = 256,
    opset: int = 17,
) -> None:
    """Export a trained UNetRadioMap to ONNX."""
    _require_torch()
    import torch

    model.eval()
    dummy = torch.randn(1, 5, grid_size, grid_size)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        verbose=False,
    )
    print(f"ONNX export: {output_path}")


__all__ = ["UNetRadioMap", "build_unet", "count_parameters", "export_to_onnx"]
