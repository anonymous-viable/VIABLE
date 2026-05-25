"""Temporal Tracking module — multi-frame detection-based motion tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .object_detector import DetectedObject


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class TemporalInfo:
    """Cross-frame motion information from object tracking."""
    object_trajectories: dict  # {label: [(x,y), (x,y), ...]}  per-frame 2D
    motion_descriptions: dict  # {label: "moving right and up"}

    def to_text(self) -> str:
        lines = ["[Temporal Motion Info]"]
        for lbl, traj in self.object_trajectories.items():
            traj_str = " -> ".join(f"({x:.0f},{y:.0f})" for x, y in traj)
            desc = self.motion_descriptions.get(lbl, "")
            lines.append(f"  {lbl}: {traj_str}  ({desc})")
        return "\n".join(lines)


# =========================================================================
#  TemporalTracker
# =========================================================================

class TemporalTracker:
    """
    Infer object motion by comparing positions across multi-frame detections.
    No extra model needed.
    """

    @staticmethod
    def _describe_motion(trajectory: List[tuple]) -> str:
        """Describe motion direction from trajectory start/end points."""
        if len(trajectory) < 2:
            return "stationary"
        x0, y0 = trajectory[0]
        x1, y1 = trajectory[-1]
        dx, dy = x1 - x0, y1 - y0
        threshold = 5.0  # pixels

        h_dir = ""
        if dx > threshold:
            h_dir = "right"
        elif dx < -threshold:
            h_dir = "left"

        v_dir = ""
        if dy > threshold:
            v_dir = "down"
        elif dy < -threshold:
            v_dir = "up"

        if h_dir and v_dir:
            return f"moving {v_dir}-{h_dir}"
        elif h_dir:
            return f"moving {h_dir}"
        elif v_dir:
            return f"moving {v_dir}"
        else:
            return "roughly stationary"

    def track(
        self,
        objects_per_frame: List[List[DetectedObject]],
    ) -> TemporalInfo:
        """
        Track object motion across frames.
        Matches objects by label across different frames.
        """
        label_positions = {}  # {label: [(cx, cy), ...]}
        for frame_objs in objects_per_frame:
            seen_this_frame = set()
            for obj in frame_objs:
                if obj.label not in label_positions:
                    label_positions[obj.label] = []
                if obj.label not in seen_this_frame:
                    if obj.centroid is not None:
                        label_positions[obj.label].append(obj.centroid)
                    else:
                        x1, y1, x2, y2 = obj.bbox
                        label_positions[obj.label].append(((x1+x2)/2, (y1+y2)/2))
                    seen_this_frame.add(obj.label)

        # Only keep labels that appear in at least 2 frames
        trajectories = {
            lbl: traj for lbl, traj in label_positions.items()
            if len(traj) >= 2
        }
        descriptions = {
            lbl: self._describe_motion(traj)
            for lbl, traj in trajectories.items()
        }

        return TemporalInfo(
            object_trajectories=trajectories,
            motion_descriptions=descriptions,
        )

    def track_to_text(
        self,
        objects_per_frame: List[List[DetectedObject]],
    ) -> str:
        info = self.track(objects_per_frame)
        return info.to_text()
