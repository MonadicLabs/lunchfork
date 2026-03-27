"""
shared.models — Inference backend wrappers for lunchfork.

Provides:
  - InferenceBackend: abstract base for ONNX/PyTorch backends
  - get_backend(): factory reading INFERENCE_BACKEND env var
  - DiffusionModel: wrapper around the diffusion radio-map model
  - SuperResolutionModel: wrapper around the SR×4 model

Phase 1/2: stubs returning plausible random outputs.
Phase 3: connect to real ONNX/PyTorch checkpoints.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class InferenceBackend(ABC):
    """Abstract inference backend (ONNX Runtime or PyTorch)."""

    @abstractmethod
    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference. Returns named output tensors."""
        ...


class _StubBackend(InferenceBackend):
    """Stub backend used when no model checkpoint is available."""

    def __init__(self, model_path: str) -> None:
        logger.warning("inference.stub_backend", model_path=model_path)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Return zeros with same spatial shape as first input
        first = next(iter(inputs.values()))
        h, w = first.shape[-2], first.shape[-1]
        return {"output": np.zeros((1, h, w), dtype=np.float32)}


class OnnxBackend(InferenceBackend):
    """ONNX Runtime backend."""

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        import onnxruntime as ort  # type: ignore[import]

        providers = providers or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)
        logger.info("onnx.backend_loaded", model_path=model_path, providers=providers)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = self._session.run(None, inputs)
        out_names = [o.name for o in self._session.get_outputs()]
        return dict(zip(out_names, outputs))


class TorchBackend(InferenceBackend):
    """PyTorch backend (CUDA or CPU)."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        import torch  # type: ignore[import]

        self._device = torch.device(device)
        self._model = torch.jit.load(model_path, map_location=self._device)
        self._model.eval()
        logger.info("torch.backend_loaded", model_path=model_path, device=device)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        import torch  # type: ignore[import]

        with torch.no_grad():
            tensors = {k: torch.from_numpy(v).to(self._device) for k, v in inputs.items()}
            # Assumes single-input single-output TorchScript model
            inp = next(iter(tensors.values()))
            out = self._model(inp)
        return {"output": out.cpu().numpy()}


def get_backend(model_path: str) -> InferenceBackend:
    """
    Return an InferenceBackend for the given model path.

    Reads INFERENCE_BACKEND env var: onnx-cpu | onnx-cuda | pytorch-cuda | pytorch-cpu
    Falls back to stub if model file does not exist.
    """
    if not Path(model_path).exists():
        logger.warning("inference.model_not_found", model_path=model_path)
        return _StubBackend(model_path)

    backend_name = os.environ.get("INFERENCE_BACKEND", "onnx-cpu").lower()

    try:
        if backend_name == "onnx-cpu":
            return OnnxBackend(model_path, providers=["CPUExecutionProvider"])
        if backend_name == "onnx-cuda":
            return OnnxBackend(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        if backend_name == "pytorch-cuda":
            return TorchBackend(model_path, device="cuda")
        if backend_name == "pytorch-cpu":
            return TorchBackend(model_path, device="cpu")
    except Exception as exc:
        logger.error(
            "inference.backend_load_failed",
            backend=backend_name,
            model_path=model_path,
            error=str(exc),
        )
        return _StubBackend(model_path)

    logger.warning("inference.unknown_backend", backend=backend_name)
    return _StubBackend(model_path)


class UNetModel:
    """
    Wrapper around the U-Net radio map completion model.

    Drop-in replacement for DiffusionModel — same interface, single forward pass
    (no diffusion steps), so inference is ~50× faster.

    Inputs:
      conditioning: [3, H, W] float32 — DEM, buildings, vegetation
      sparse_rssi:  list of (x_px, y_px, rssi_norm) tuples

    Output: [1, H, W] float32 radio map in [0, 1]
    """

    def __init__(self, model_path: str) -> None:
        self._backend = get_backend(model_path)
        self._model_path = model_path

    def infer(
        self,
        conditioning: np.ndarray,
        sparse_rssi: list[tuple[float, float, float]],
    ) -> np.ndarray:
        """
        Run U-Net inference.

        conditioning: [3, H, W]
        sparse_rssi:  [(x_px, y_px, rssi_norm), ...]
        Returns:      [1, H, W]
        """
        h, w = conditioning.shape[1], conditioning.shape[2]

        rssi_map = np.zeros((1, h, w), dtype=np.float32)
        rssi_mask = np.zeros((1, h, w), dtype=np.float32)
        for x_px, y_px, rssi_norm in sparse_rssi:
            xi, yi = int(round(x_px)), int(round(y_px))
            if 0 <= xi < w and 0 <= yi < h:
                rssi_map[0, yi, xi] = rssi_norm
                rssi_mask[0, yi, xi] = 1.0

        # [1, 5, H, W]
        model_input = np.concatenate(
            [conditioning, rssi_map, rssi_mask], axis=0
        )[np.newaxis].astype(np.float32)

        outputs = self._backend.run({"input": model_input})
        radiomap = outputs.get("output", np.zeros((1, 1, h, w), dtype=np.float32))

        if radiomap.ndim == 4:
            radiomap = radiomap[0]

        return radiomap  # [1, H, W]


class DiffusionModel:
    """
    Wrapper around the conditional diffusion radio-map model.

    Inputs:
      conditioning: [3, H, W] float32 — DEM, buildings, vegetation
      sparse_rssi: list of (x_px, y_px, rssi_norm) tuples

    Output: [1, H, W] float32 radio map (normalised path loss / signal likelihood)
    """

    def __init__(self, model_path: str) -> None:
        self._backend = get_backend(model_path)
        self._model_path = model_path

    def infer(
        self,
        conditioning: np.ndarray,
        sparse_rssi: list[tuple[float, float, float]],
        n_steps: int = 50,
    ) -> np.ndarray:
        """
        Run diffusion inference.

        conditioning: [3, H, W]
        sparse_rssi: [(x_px, y_px, rssi_norm), ...]
        Returns: [1, H, W] radio map
        """
        h, w = conditioning.shape[1], conditioning.shape[2]

        # Build sparse RSSI map [1, H, W]: zero everywhere, fill at observation locations
        rssi_map = np.zeros((1, h, w), dtype=np.float32)
        rssi_mask = np.zeros((1, h, w), dtype=np.float32)
        for x_px, y_px, rssi_norm in sparse_rssi:
            xi, yi = int(round(x_px)), int(round(y_px))
            if 0 <= xi < w and 0 <= yi < h:
                rssi_map[0, yi, xi] = rssi_norm
                rssi_mask[0, yi, xi] = 1.0

        # Concatenate: [3+1+1, H, W]
        model_input = np.concatenate(
            [conditioning, rssi_map, rssi_mask], axis=0
        )[np.newaxis, ...]  # [1, 5, H, W]

        outputs = self._backend.run({"input": model_input.astype(np.float32)})
        radiomap = outputs.get("output", np.zeros((1, 1, h, w), dtype=np.float32))

        # Remove batch dimension if present
        if radiomap.ndim == 4:
            radiomap = radiomap[0]

        return radiomap  # [1, H, W]


class SuperResolutionModel:
    """
    Wrapper around the SR×4 super-resolution model.

    Inputs:
      radiomap_lr: [1, H, W] float32
      mnt_hr: [1, H*SR, W*SR] float32

    Output: [1, H*SR, W*SR] float32
    """

    def __init__(self, model_path: str) -> None:
        self._backend = get_backend(model_path)

    def upscale(
        self,
        radiomap_lr: np.ndarray,
        mnt_hr: np.ndarray,
    ) -> np.ndarray:
        """
        Upscale radiomap_lr conditioned on high-res DEM mnt_hr.

        Returns [1, H*SR, W*SR] float32.
        """
        h_hr, w_hr = mnt_hr.shape[1], mnt_hr.shape[2]

        model_input = np.concatenate(
            [radiomap_lr, mnt_hr], axis=0
        )[np.newaxis, ...]  # [1, 2, H_hr, W_hr]  (after interpolating lr to hr size)

        # Upsample lr to hr size for concatenation
        from shared.geo import _resize_bilinear
        lr_up = _resize_bilinear(radiomap_lr[0], h_hr, w_hr)[np.newaxis, np.newaxis, ...]
        # lr_up: [1, 1, H_hr, W_hr]

        model_input = np.concatenate(
            [lr_up[:, 0, :, :], mnt_hr[np.newaxis, 0, :, :]], axis=0
        )[np.newaxis, ...]  # [1, 2, H_hr, W_hr]

        outputs = self._backend.run({"input": model_input.astype(np.float32)})
        out = outputs.get("output", np.zeros((1, 1, h_hr, w_hr), dtype=np.float32))

        if out.ndim == 4:
            out = out[0]

        return out  # [1, H_hr, W_hr]


__all__ = [
    "InferenceBackend",
    "OnnxBackend",
    "TorchBackend",
    "get_backend",
    "UNetModel",
    "DiffusionModel",
    "SuperResolutionModel",
]
