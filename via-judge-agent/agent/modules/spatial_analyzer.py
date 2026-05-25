"""Spatial Analysis module — detection + depth based spatial reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .object_detector import DetectedObject


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SpatialInfo:
    """3D spatial information derived from detection + depth."""
    camera_poses: list          # per-frame (R, t)
    object_positions_3d: dict   # {label: (x, y, z)}
    pairwise_distances: dict    # {(label_a, label_b): distance}

    def to_text(self) -> str:
        lines = ["[Spatial 3D Info]"]
        # Camera motion
        if len(self.camera_poses) >= 2:
            t0 = np.array(self.camera_poses[0][1])
            t1 = np.array(self.camera_poses[-1][1])
            delta = t1 - t0
            lines.append(
                f"  Camera translation (first->last): "
                f"dx={delta[0]:.3f}, dy={delta[1]:.3f}, dz={delta[2]:.3f}"
            )
        # Object 3D positions
        for lbl, pos in self.object_positions_3d.items():
            lines.append(f"  {lbl}: 3d_pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        # Pairwise
        for (a, b), dist in self.pairwise_distances.items():
            lines.append(f"  dist({a}, {b}) = {dist:.2f}")
        return "\n".join(lines)


# =========================================================================
#  SpatialAnalyzer
# =========================================================================

class SpatialAnalyzer:
    """
    Infer spatial relationships from multi-frame detections + depth.
    No extra model needed — uses ObjectDetector and DepthEstimator outputs directly.
    """

    def analyze(
        self,
        objects_per_frame: List[List[DetectedObject]],
        depth_per_frame: Optional[List[dict]] = None,
    ) -> SpatialInfo:
        """
        Infer spatial relationships from detection + depth results.

        Args:
            objects_per_frame: detection results per frame
            depth_per_frame: {label: depth_value} per frame

        Returns:
            SpatialInfo
        """
        object_positions_3d = {}
        if objects_per_frame and objects_per_frame[0]:
            frame0_objs = objects_per_frame[0]
            for obj in frame0_objs:
                if obj.centroid is not None:
                    cx, cy = obj.centroid
                else:
                    x1, y1, x2, y2 = obj.bbox
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                z = 0.0
                if depth_per_frame and obj.label in depth_per_frame[0]:
                    z = depth_per_frame[0][obj.label]
                object_positions_3d[obj.label] = (cx, cy, z)

        # Pairwise distances (2D + depth)
        pairwise_distances = {}
        labels = list(object_positions_3d.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                pa = np.array(object_positions_3d[a])
                pb = np.array(object_positions_3d[b])
                dist = float(np.linalg.norm(pa - pb))
                pairwise_distances[(a, b)] = dist

        return SpatialInfo(
            camera_poses=[],
            object_positions_3d=object_positions_3d,
            pairwise_distances=pairwise_distances,
        )

    def analyze_to_text(
        self,
        objects_per_frame: List[List[DetectedObject]],
        depth_per_frame: Optional[List[dict]] = None,
    ) -> str:
        info = self.analyze(objects_per_frame, depth_per_frame)
        return info.to_text()
