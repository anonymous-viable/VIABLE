"""Depth Estimation module — Depth Anything V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from .object_detector import DetectedObject


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DepthInfo:
    """Depth info for a single frame."""
    frame_index: int
    object_depths: dict  # {label: depth_value}
    depth_ranking: list  # labels sorted near-to-far

    def to_text(self) -> str:
        ranking_str = " < ".join(
            f"{lbl}({d:.2f})" for lbl, d in
            sorted(self.object_depths.items(), key=lambda x: x[1])
        )
        return (
            f"Frame {self.frame_index} depth (near->far): {ranking_str}"
        )


# =========================================================================
#  DepthEstimator — Depth Anything V3
# =========================================================================

class DepthEstimator:
    """
    Depth Anything V3 (DA3) monocular depth estimation.

    Supports single and multi-frame input. Multi-frame also outputs camera poses.
    Core flow: Object Detector centroids -> sample depth map -> output text.
    """

    def __init__(
        self,
        model_id: str = None,
        device: str = "cuda:0",
    ):
        from agent.config.run_config import DEPTH_MODEL_PATH
        self.device = device
        self._model_id = model_id or DEPTH_MODEL_PATH
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from depth_anything_3.api import DepthAnything3

        self._model = DepthAnything3.from_pretrained(self._model_id)
        self._model = self._model.to(
            device=torch.device(self.device if torch.cuda.is_available() else "cpu")
        )

    def estimate(
        self,
        images: Union[str, List[str]],
    ):
        """
        Run DA3 inference.

        Args:
            images: single image path (str) or list of paths

        Returns:
            prediction object:
              - prediction.depth: np.ndarray [N, H, W] depth maps
              - prediction.conf: np.ndarray [N, H, W] confidence
              - prediction.extrinsics: np.ndarray [N, 3, 4] camera extrinsics (multi-frame)
              - prediction.intrinsics: np.ndarray [N, 3, 3] camera intrinsics
        """
        self._ensure_loaded()
        if isinstance(images, str):
            images = [images]
        prediction = self._model.inference(images)
        return prediction

    def estimate_at_objects(
        self,
        images: Union[str, List[str]],
        objects: List[DetectedObject],
        frame_index: int = 0,
    ) -> DepthInfo:
        """
        Core method: detection results + depth map -> per-object depth values.
        """
        prediction = self.estimate(images)
        depth_map = prediction.depth[frame_index]  # (H, W)
        h, w = depth_map.shape

        object_depths = {}
        for obj in objects:
            if obj.centroid is not None:
                cx, cy = obj.centroid
            else:
                x1, y1, x2, y2 = obj.bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            px = int(np.clip(cx, 0, w - 1))
            py = int(np.clip(cy, 0, h - 1))
            object_depths[obj.label] = float(depth_map[py, px])

        ranking = sorted(object_depths.keys(), key=lambda k: object_depths[k])

        return DepthInfo(
            frame_index=frame_index,
            object_depths=object_depths,
            depth_ranking=ranking,
        )

    def estimate_to_text(
        self,
        images: Union[str, List[str]],
        objects: List[DetectedObject],
        frame_index: int = 0,
    ) -> str:
        """Full pipeline: detect objects -> sample depth -> output structured text."""
        prediction = self.estimate(images)
        depth_map = prediction.depth[frame_index]
        h, w = depth_map.shape

        object_depths = {}
        for obj in objects:
            if obj.centroid is not None:
                cx, cy = obj.centroid
            else:
                x1, y1, x2, y2 = obj.bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            px = int(np.clip(cx, 0, w - 1))
            py = int(np.clip(cy, 0, h - 1))
            object_depths[obj.label] = float(depth_map[py, px])

        sorted_items = sorted(object_depths.items(), key=lambda kv: kv[1])

        lines = ["[Depth Evidence (DA3)]"]

        obj_strs = []
        for i, (lbl, d) in enumerate(sorted_items):
            tag = ""
            if i == 0:
                tag = ", nearest"
            elif i == len(sorted_items) - 1:
                tag = ", farthest"
            obj_strs.append(f"{lbl}(depth={d:.2f}{tag})")
        lines.append(f"  Frame {frame_index}: " + ", ".join(obj_strs))

        ranking_str = " < ".join(lbl for lbl, _ in sorted_items)
        lines.append(f"  Ranking (near->far): {ranking_str}")

        if isinstance(images, list) and len(images) > 1:
            if prediction.extrinsics is not None:
                t0 = prediction.extrinsics[0, :3, 3]
                t_last = prediction.extrinsics[-1, :3, 3]
                delta = t_last - t0
                lines.append(
                    f"  Camera motion (frame 0->{len(images)-1}): "
                    f"dx={delta[0]:.4f}, dy={delta[1]:.4f}, dz={delta[2]:.4f}"
                )

        return "\n".join(lines)
