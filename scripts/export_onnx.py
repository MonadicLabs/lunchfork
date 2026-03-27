#!/usr/bin/env python3
"""
scripts/export_onnx.py — Export PyTorch models to ONNX format.

Usage:
  python scripts/export_onnx.py --model diffusion \\
    --checkpoint data/checkpoints/diffusion_vhf_v1.pt \\
    --output data/checkpoints/diffusion_vhf_v1.onnx --opset 17

  python scripts/export_onnx.py --benchmark \\
    --checkpoint data/checkpoints/diffusion_vhf_v1.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export and benchmark ONNX models")
    p.add_argument("--model", choices=["diffusion", "superres", "unet"], help="Model type to export")
    p.add_argument("--checkpoint", type=Path, required=True, help="Input .pt or .onnx checkpoint")
    p.add_argument("--output", type=Path, help="Output .onnx path (export mode)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--benchmark", action="store_true", help="Benchmark ONNX model latency")
    p.add_argument("--grid-size", type=int, default=256, help="Grid resolution for benchmarking")
    p.add_argument("--n-runs", type=int, default=10, help="Number of benchmark runs")
    return p.parse_args()


def export_diffusion(checkpoint: Path, output: Path, opset: int) -> None:
    """Export diffusion model from .pt to .onnx."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required for export. Install with: pip install torch")
        sys.exit(1)

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    print(f"Loading diffusion checkpoint: {checkpoint}")
    # Load model — assumes TorchScript or state_dict
    try:
        model = torch.jit.load(str(checkpoint), map_location="cpu")
    except Exception:
        print("WARN: Not a TorchScript model — trying state_dict load (stub)")
        # Placeholder: would load architecture + weights here
        print("ERROR: Provide a TorchScript .pt file for export")
        sys.exit(1)

    model.eval()

    # Dummy input: [1, 5, H, W] — conditioning (3) + rssi_map (1) + mask (1)
    H = W = 256
    dummy_input = torch.randn(1, 5, H, W)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to: {output} (opset {opset})")

    torch.onnx.export(
        model,
        dummy_input,
        str(output),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        verbose=False,
    )
    print(f"Export complete: {output}")


def export_superres(checkpoint: Path, output: Path, opset: int) -> None:
    """Export super-resolution model from .pt to .onnx."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required for export.")
        sys.exit(1)

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    print(f"Loading SR checkpoint: {checkpoint}")
    try:
        model = torch.jit.load(str(checkpoint), map_location="cpu")
    except Exception:
        print("ERROR: Provide a TorchScript .pt file for export")
        sys.exit(1)

    model.eval()

    # Dummy input: [1, 2, H*4, W*4] — upsampled LR + HR DEM
    H = W = 256
    SR = 4
    dummy_input = torch.randn(1, 2, H * SR, W * SR)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to: {output} (opset {opset})")

    torch.onnx.export(
        model,
        dummy_input,
        str(output),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        verbose=False,
    )
    print(f"Export complete: {output}")


def export_unet(checkpoint: Path, output: Path, opset: int, grid_size: int) -> None:
    """Export U-Net from .pt checkpoint to ONNX."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required for export. pip install torch")
        sys.exit(1)

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    sys.path.insert(0, str(ROOT))
    from shared.models.unet_arch import UNetRadioMap, export_to_onnx

    print(f"Loading U-Net checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        cfg   = ckpt.get("args", {})
        base  = cfg.get("base", 32)
        in_ch = cfg.get("in_ch", 5)
        model = UNetRadioMap(in_ch=in_ch, base=base)
        model.load_state_dict(ckpt["model_state"])
        print(f"  base={base}  in_ch={in_ch}  epoch={ckpt.get('epoch', '?')}")
    else:
        # Bare state_dict — assume defaults
        model = UNetRadioMap(in_ch=5, base=32)
        model.load_state_dict(ckpt)

    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(model, str(output), grid_size=grid_size, opset=opset)
    print(f"Export complete: {output}")


def benchmark_onnx(checkpoint: Path, grid_size: int, n_runs: int) -> None:
    """Benchmark ONNX model latency."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("ERROR: onnxruntime required. Install with: pip install onnxruntime")
        sys.exit(1)

    if not checkpoint.exists():
        print(f"ERROR: Model not found: {checkpoint}")
        sys.exit(1)

    print(f"Loading ONNX model: {checkpoint}")
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(checkpoint), providers=providers)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # Build test input
    H = W = grid_size
    dummy_input = np.random.randn(1, 5, H, W).astype(np.float32)

    print(f"Benchmarking {n_runs} runs at {grid_size}×{grid_size}...")

    # Warm up
    for _ in range(2):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        print(f"  Run {i+1}: {dt:.1f}ms")

    import statistics
    print(f"\nResults ({n_runs} runs @ {grid_size}×{grid_size}):")
    print(f"  Mean:   {statistics.mean(times):.1f}ms")
    print(f"  Median: {statistics.median(times):.1f}ms")
    print(f"  Stdev:  {statistics.stdev(times):.1f}ms")
    print(f"  Min:    {min(times):.1f}ms")
    print(f"  Max:    {max(times):.1f}ms")


def main() -> None:
    args = parse_args()

    if args.benchmark:
        benchmark_onnx(args.checkpoint, args.grid_size, args.n_runs)
        return

    if args.model is None:
        print("ERROR: --model required for export mode")
        sys.exit(1)

    if args.output is None:
        print("ERROR: --output required for export mode")
        sys.exit(1)

    if args.model == "diffusion":
        export_diffusion(args.checkpoint, args.output, args.opset)
    elif args.model == "superres":
        export_superres(args.checkpoint, args.output, args.opset)
    elif args.model == "unet":
        export_unet(args.checkpoint, args.output, args.opset, args.grid_size)


if __name__ == "__main__":
    main()
