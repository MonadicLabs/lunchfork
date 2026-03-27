#!/usr/bin/env python3
"""
training/train_superres.py — Train SR×4 super-resolution model for radio maps.

Input:  [B, 2, H*SR, W*SR]  — upsampled LR radio map + HR DEM
Output: [B, 1, H*SR, W*SR]  — refined HR radio map

Usage:
  python training/train_superres.py \\
    --dataset data/datasets/vhf_uhf_outdoor_v1/ \\
    --sr-factor 4 \\
    --output data/checkpoints/sr_vhf_v1.pt \\
    --epochs 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train lunchfork SR model")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--sr-factor", type=int, default=4)
    p.add_argument("--output", type=Path, default=ROOT / "data/checkpoints/sr_vhf_v1.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-every", type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SRDataset:
    """
    Dataset for super-resolution.

    From each scene:
      LR:  bicubic downsample of GT radio map → SR×4 upsample back to original size
      HR_DEM: high-res DEM channel (1st channel of conditioning)
      Target: original GT radio map (HR)
    """

    def __init__(self, dataset_dir: Path, sr_factor: int = 4) -> None:
        self._files = sorted(dataset_dir.glob("scene_*.npz"))
        self._sr = sr_factor
        if not self._files:
            raise FileNotFoundError(f"No scene_*.npz files in {dataset_dir}")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        data = np.load(self._files[idx])
        radiomap_hr = data["radiomap_gt"]   # [H, W] — HR ground truth
        conditioning = data["conditioning"] # [3, H, W]

        H, W = radiomap_hr.shape

        # HR DEM from conditioning channel 0
        dem_hr = conditioning[0]  # [H, W]

        # Simulate LR input: downsample by SR then upsample back
        lr_size = (H // self._sr, W // self._sr)
        radiomap_lr = _resize_bilinear_np(radiomap_hr, lr_size[0], lr_size[1])
        radiomap_lr_up = _resize_bilinear_np(radiomap_lr, H, W)  # [H, W]

        # Model input: [2, H, W]
        model_input = np.stack([radiomap_lr_up, dem_hr], axis=0).astype(np.float32)

        return {
            "input": model_input,
            "target": radiomap_hr[np.newaxis, :, :].astype(np.float32),  # [1, H, W]
        }


def _resize_bilinear_np(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Simple bilinear resize (no scipy dependency)."""
    in_h, in_w = arr.shape
    row_idx = np.linspace(0, in_h - 1, out_h)
    col_idx = np.linspace(0, in_w - 1, out_w)
    r0 = np.floor(row_idx).astype(int).clip(0, in_h - 2)
    c0 = np.floor(col_idx).astype(int).clip(0, in_w - 2)
    r1 = r0 + 1
    c1 = c0 + 1
    dr = (row_idx - r0)[:, np.newaxis]
    dc = (col_idx - c0)[np.newaxis, :]
    return (
        arr[r0][:, c0] * (1 - dr) * (1 - dc)
        + arr[r0][:, c1] * (1 - dr) * dc
        + arr[r1][:, c0] * dr * (1 - dc)
        + arr[r1][:, c1] * dr * dc
    ).astype(np.float32)


def collate_fn(batch: list[dict]) -> dict[str, "torch.Tensor"]:
    import torch
    inputs = torch.from_numpy(np.stack([b["input"] for b in batch]))
    targets = torch.from_numpy(np.stack([b["target"] for b in batch]))
    return {"input": inputs, "target": targets}


# ---------------------------------------------------------------------------
# Model — Residual Channel Attention Network (RCAN-style)
# ---------------------------------------------------------------------------


def build_sr_model(sr_factor: int = 4) -> "torch.nn.Module":
    """
    Simple SR model: conv + residual blocks + pixel shuffle upsampling.
    Input: [B, 2, H, W]
    Output: [B, 1, H, W] (H, W already at HR resolution when using pre-upsampled input)
    """
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(min(8, channels), channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(min(8, channels), channels),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.block(x)

    class SRNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            C = 64
            self.head = nn.Conv2d(2, C, 3, padding=1)
            self.body = nn.Sequential(*[ResBlock(C) for _ in range(8)])
            self.tail = nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(C, 1, 3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            feat = self.head(x)
            feat = feat + self.body(feat)  # global residual
            return self.tail(feat)

    return SRNet()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("ERROR: PyTorch required for training.")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    print(f"Device: {device}")

    print(f"Loading dataset: {args.dataset}")
    dataset = SRDataset(args.dataset, sr_factor=args.sr_factor)
    print(f"  {len(dataset)} scenes, SR×{args.sr_factor}")

    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    from torch.utils.data import DataLoader, Subset

    train_loader = DataLoader(
        Subset(dataset, indices[:n_train]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(dataset, indices[n_train:]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    print(f"  Train: {n_train} | Val: {n_val}")

    model = build_sr_model(sr_factor=args.sr_factor).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    # Combined L1 + perceptual (MSE proxy) loss
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inp = batch["input"].to(device)
            tgt = batch["target"].to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = 0.8 * l1_loss(pred, tgt) + 0.2 * mse_loss(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input"].to(device)
                tgt = batch["target"].to(device)
                pred = model(inp)
                val_loss += l1_loss(pred, tgt).item()
        val_loss /= max(len(val_loader), 1)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss, "lr": lr_now})
        print(f"Epoch {epoch:4d}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f} | lr={lr_now:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                str(args.output),
            )
            print(f"  Saved → {args.output}")

        if epoch % args.save_every == 0:
            path = args.output.parent / f"{args.output.stem}_ep{epoch}.pt"
            torch.save(model.state_dict(), str(path))

    hist_path = args.output.parent / f"{args.output.stem}_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"\nDone. Best val: {best_val:.4f} | Checkpoint: {args.output}")


if __name__ == "__main__":
    train(parse_args())
