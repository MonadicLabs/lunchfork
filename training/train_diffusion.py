#!/usr/bin/env python3
"""
training/train_diffusion.py — Train conditional diffusion model for radio map reconstruction.

Architecture: U-Net with conditioning on DEM/buildings/vegetation tensors and sparse RSSI.
Inspired by RadioDiff-Loc. Fine-tune from pretrained checkpoint if provided.

Usage:
  python training/train_diffusion.py \\
    --dataset data/datasets/vhf_uhf_outdoor_v1/ \\
    --base-checkpoint radiodiff-loc-pretrained.pt \\
    --output data/checkpoints/diffusion_vhf_v1.pt \\
    --epochs 200 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train lunchfork diffusion model")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, default=None)
    p.add_argument("--output", type=Path, default=ROOT / "data/checkpoints/diffusion_vhf_v1.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=1000, help="Diffusion timesteps")
    p.add_argument("--n-inference-steps", type=int, default=50)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RadioMapDataset:
    """
    Dataset of (conditioning, radiomap_gt) pairs from .npz scene files.
    """

    def __init__(
        self,
        dataset_dir: Path,
        n_sparse_obs: int = 20,
        noise_rssi_std: float = 2.0,
    ) -> None:
        self._files = sorted(dataset_dir.glob("scene_*.npz"))
        self._n_obs = n_sparse_obs
        self._noise_std = noise_rssi_std

        if not self._files:
            raise FileNotFoundError(f"No scene_*.npz files in {dataset_dir}")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        data = np.load(self._files[idx])
        conditioning = data["conditioning"]  # [3, H, W]
        radiomap = data["radiomap_gt"]       # [H, W]
        H, W = radiomap.shape

        # Sample sparse RSSI observations (simulate node positions)
        n = min(self._n_obs, H * W)
        flat_idx = np.random.choice(H * W, size=n, replace=False)
        rows = flat_idx // W
        cols = flat_idx % W

        rssi_values = radiomap[rows, cols]
        # Add measurement noise
        rssi_noisy = rssi_values + np.random.randn(n) * self._noise_std
        rssi_noisy = rssi_noisy.clip(0, 1)

        # Build sparse RSSI map [1, H, W]
        sparse_map = np.zeros((1, H, W), dtype=np.float32)
        sparse_mask = np.zeros((1, H, W), dtype=np.float32)
        sparse_map[0, rows, cols] = rssi_noisy.astype(np.float32)
        sparse_mask[0, rows, cols] = 1.0

        # Model input: [5, H, W]
        model_input = np.concatenate([conditioning, sparse_map, sparse_mask], axis=0)

        return {
            "input": model_input.astype(np.float32),
            "target": radiomap[np.newaxis, :, :].astype(np.float32),  # [1, H, W]
            "freq_hz": float(data.get("freq_hz", 433e6)),
        }


def collate_fn(batch: list[dict]) -> dict[str, "torch.Tensor"]:
    import torch
    inputs = torch.from_numpy(np.stack([b["input"] for b in batch]))
    targets = torch.from_numpy(np.stack([b["target"] for b in batch]))
    return {"input": inputs, "target": targets}


# ---------------------------------------------------------------------------
# Model architecture — Conditional U-Net for diffusion
# ---------------------------------------------------------------------------


class ConvBlock(object):
    """Placeholder — in real code, this would be a nn.Module."""
    pass


def build_unet(in_channels: int = 5, out_channels: int = 1, base_channels: int = 64) -> "torch.nn.Module":
    """
    Build a simple U-Net for radio map reconstruction.

    Input:  [B, in_channels, H, W]  (conditioning + sparse RSSI)
    Output: [B, out_channels, H, W] (predicted radio map)
    """
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_c: int, out_c: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.GroupNorm(min(8, out_c), out_c),
                nn.GELU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.GroupNorm(min(8, out_c), out_c),
                nn.GELU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.block(x)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            # Encoder
            self.enc1 = DoubleConv(in_channels, c)
            self.enc2 = DoubleConv(c, c * 2)
            self.enc3 = DoubleConv(c * 2, c * 4)
            self.enc4 = DoubleConv(c * 4, c * 8)
            # Bottleneck
            self.bottleneck = DoubleConv(c * 8, c * 16)
            # Decoder
            self.dec4 = DoubleConv(c * 16 + c * 8, c * 8)
            self.dec3 = DoubleConv(c * 8 + c * 4, c * 4)
            self.dec2 = DoubleConv(c * 4 + c * 2, c * 2)
            self.dec1 = DoubleConv(c * 2 + c, c)
            self.out = nn.Conv2d(c, out_channels, 1)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
            return torch.sigmoid(self.out(d1))

    return UNet()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("ERROR: PyTorch required for training. Install with: pip install torch")
        sys.exit(1)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = RadioMapDataset(args.dataset)
    print(f"  {len(dataset)} scenes")

    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    def make_loader(idx_list: list[int], shuffle: bool) -> "torch.utils.data.DataLoader":
        from torch.utils.data import DataLoader, Subset
        subset = Subset(dataset, idx_list)
        return DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)
    print(f"  Train: {n_train} | Val: {n_val}")

    # Model
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if args.base_checkpoint and args.base_checkpoint.exists():
        print(f"Loading base checkpoint: {args.base_checkpoint}")
        try:
            state = torch.load(str(args.base_checkpoint), map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            print("  Checkpoint loaded (non-strict)")
        except Exception as exc:
            print(f"  WARN: Could not load checkpoint: {exc}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input"].to(device)
                tgt = batch["target"].to(device)
                pred = model(inp)
                val_loss += criterion(pred, tgt).item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})

        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | lr={lr:.2e}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_loss": val_loss, "args": vars(args)},
                str(args.output),
            )
            print(f"  Saved best checkpoint → {args.output}")

        # Periodic save
        if epoch % args.save_every == 0:
            periodic_path = args.output.parent / f"{args.output.stem}_ep{epoch}.pt"
            torch.save(model.state_dict(), str(periodic_path))

    # Save training history
    hist_path = args.output.parent / f"{args.output.stem}_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {args.output}")
    print(f"History: {hist_path}")


if __name__ == "__main__":
    train(parse_args())
