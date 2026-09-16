"""MiDaS-small monocular depth estimation, used by the Phase 4 occlusion grid
as the camera-only "is this space clear?" signal.

Deliberately separate from CARLA's synthetic ground-truth depth camera (used
only for dataset labeling in carla_tools.occlusion_mask): the point of this
module is to mimic what a real camera-only system could infer at runtime, and
a monocular depth network is the realistic proxy for that per the spec.

Important limitation, documented rather than hidden: MiDaS outputs *relative
inverse depth* (disparity-like, no fixed physical scale, and only consistent
within a single frame) -- not metric meters. `is_clear_at` therefore compares
each cell's disparity to a per-frame-normalized threshold rather than a
metric distance. This is a real, well-known limitation of monocular depth
estimation (scale ambiguity), and is itself part of why the pipeline fuses in
radar rather than relying on camera depth alone.
"""
import cv2
import numpy as np
import torch

_MODEL = None
_TRANSFORM = None
_DEVICE = None


def load_midas(device: str | None = None):
    global _MODEL, _TRANSFORM, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TRANSFORM, _DEVICE

    _DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    _MODEL.to(_DEVICE).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    _TRANSFORM = transforms.small_transform
    return _MODEL, _TRANSFORM, _DEVICE


def estimate_disparity(rgb: np.ndarray) -> np.ndarray:
    """rgb: HxWx3 uint8 RGB array (as returned by carla_tools.sensors.rgb_to_array).
    Returns an HxW float32 disparity map, same size as input -- higher values
    mean closer, per MiDaS convention. Not metric; see module docstring."""
    model, transform, device = load_midas()
    input_batch = transform(rgb).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()


def normalize_disparity(disparity: np.ndarray) -> np.ndarray:
    """Per-frame min-max normalize to [0, 1] so a fixed threshold in
    `is_clear_at` is meaningful despite MiDaS's frame-to-frame scale drift."""
    d_min, d_max = disparity.min(), disparity.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(disparity)
    return (disparity - d_min) / (d_max - d_min)


def is_clear_at(norm_disparity: np.ndarray, px: int, py: int, clear_threshold: float = 0.5) -> bool:
    """True if the pixel looks relatively "far" (below threshold) in this
    frame's normalized disparity -- i.e. no nearby surface breaking up what
    should be open road/space at that point."""
    h, w = norm_disparity.shape
    if not (0 <= px < w and 0 <= py < h):
        return False
    return bool(norm_disparity[py, px] < clear_threshold)
