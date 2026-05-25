"""
Visual Evidence Extractor — orchestrates all sub-modules to produce
a unified textual evidence string for the judge agent.

Refactored from the legacy visual_evidence_extractor.py into a clean
composition of ObjectDetector, DepthEstimator, SpatialAnalyzer,
TemporalTracker, and HandTracker.
"""

from __future__ import annotations

from typing import List, Optional, Union

from PIL import Image

from agent.config.run_config import (
    DEVICE,
    GROUNDING_DINO_PATH,
    SAM2_CHECKPOINT,
    DEPTH_MODEL_PATH,
)
from agent.config.vis_config import get_detection_prompts
from agent.modules.object_detector import (
    ObjectDetector,
    DetectedObject,
    _compute_iou,
)
from agent.modules.depth_estimator import DepthEstimator, DepthInfo
from agent.modules.spatial_analyzer import SpatialAnalyzer
from agent.modules.temporal_tracker import TemporalTracker
from agent.modules.hand_tracker import TrajectoryBuilder, match_hands_objects, distinguish_hands


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class VisualEvidenceExtractor:
    """
    Orchestrates detection, depth, spatial, temporal, and hand-tracking
    modules to produce a single evidence text string for the judge.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.detector = ObjectDetector(
            grounding_model_id=GROUNDING_DINO_PATH,
            sam2_checkpoint=SAM2_CHECKPOINT,
            device=device,
        )
        self.depth_estimator = DepthEstimator(
            model_id=DEPTH_MODEL_PATH,
            device=device,
        )
        self.spatial_analyzer = SpatialAnalyzer()
        self.temporal_tracker = TemporalTracker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        images: List[Union[str, Image.Image]],
        text_prompt: Union[str, List[str]] = "",
        query_points: Optional[dict] = None,
        task: str = "",
    ) -> str:
        """
        Run the full evidence extraction pipeline.

        Args:
            images: List of frame paths or PIL images.
            text_prompt: Optional explicit detection prompt(s).
            query_points: Reserved for future point-tracking use.
            task: Task identifier (e.g. "via_egodex") used to select
                  task-specific detection prompts.

        Returns:
            Concatenated textual evidence from all modules.
        """
        if not images:
            return ""

        # 1. Determine detection prompts
        prompts = get_detection_prompts(task, batch_size=len(images))
        if text_prompt:
            # If an explicit prompt is provided, prepend it
            if isinstance(text_prompt, list):
                prompts = text_prompt + prompts
            else:
                prompts = [text_prompt] + prompts

        # 2. Per-frame object detection
        objects_per_frame: List[List[DetectedObject]] = []
        for idx, img in enumerate(images):
            frame_objects: List[DetectedObject] = []
            for prompt in prompts:
                detections = self.detector.detect(image=img, text_prompt=prompt)
                frame_objects.extend(detections)
            # Deduplicate within frame
            frame_objects = self._deduplicate(frame_objects)
            objects_per_frame.append(frame_objects)

        # 3. Per-frame depth estimation
        depth_per_frame: List[dict] = []
        depth_texts: List[str] = []
        for idx, img in enumerate(images):
            if objects_per_frame[idx]:
                depth_info = self.depth_estimator.estimate_at_objects(
                    images=img if isinstance(img, str) else [img],
                    objects=objects_per_frame[idx],
                    frame_index=0,
                )
                depth_per_frame.append(depth_info.object_depths)
                depth_texts.append(depth_info.to_text())
            else:
                depth_per_frame.append({})
                depth_texts.append("")

        # 4. Spatial analysis (requires >= 2 frames)
        spatial_text = ""
        if len(images) >= 2 and any(objs for objs in objects_per_frame):
            spatial_info = self.spatial_analyzer.analyze(
                objects_per_frame=objects_per_frame,
                depth_per_frame=depth_per_frame,
            )
            spatial_text = spatial_info.to_text()

        # 5. Temporal tracking (requires >= 2 frames)
        temporal_text = ""
        if len(images) >= 2 and any(objs for objs in objects_per_frame):
            temporal_info = self.temporal_tracker.track(
                objects_per_frame=objects_per_frame,
            )
            temporal_text = temporal_info.to_text()

        # 6. Hand tracking (via_egodex task only)
        hand_text = ""
        if task == "via_egodex":
            hand_text = self._extract_hand_trajectories(images)

        # 7. Assemble final evidence string
        sections = []

        # Detection summary
        det_lines = ["[Object Detection]"]
        for idx, objs in enumerate(objects_per_frame):
            if objs:
                obj_strs = [
                    f"{o.label}({o.confidence:.2f})" for o in objs
                ]
                det_lines.append(f"  Frame {idx}: {', '.join(obj_strs)}")
        if len(det_lines) > 1:
            sections.append("\n".join(det_lines))

        # Depth
        for dt in depth_texts:
            if dt:
                sections.append(dt)
                break  # Include first non-empty depth text as representative

        # Spatial
        if spatial_text:
            sections.append(spatial_text)

        # Temporal
        if temporal_text:
            sections.append(temporal_text)

        # Hand tracking
        if hand_text:
            sections.append(hand_text)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(
        objects: List[DetectedObject],
        iou_threshold: float = 0.5,
    ) -> List[DetectedObject]:
        """
        Remove duplicate detections by IoU. When two boxes overlap above
        the threshold, keep the one with higher confidence.
        """
        if not objects:
            return []

        # Sort by confidence descending so we keep the best ones
        sorted_objs = sorted(objects, key=lambda o: o.confidence, reverse=True)
        kept: List[DetectedObject] = []

        for obj in sorted_objs:
            is_duplicate = False
            for existing in kept:
                if _compute_iou(obj.bbox, existing.bbox) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(obj)

        return kept

    def _extract_hand_trajectories(
        self,
        images: List[Union[str, Image.Image]],
    ) -> str:
        """
        Run hand-object trajectory extraction for egocentric tasks.
        Uses the HandTracker (TrajectoryBuilder) from hand_tracker module.
        """
        # Detect hands in each frame and build trajectories
        hand_prompt = "hand."
        builder = TrajectoryBuilder()

        for img in images:
            # Detect all objects including hands
            detections = self.detector.detect(image=img, text_prompt=hand_prompt)
            hands = [d for d in detections if "hand" in d.label.lower()]
            objects = [d for d in detections if "hand" not in d.label.lower()]

            # Also include previously detected objects for matching
            # Re-detect with broader prompt for context
            all_detections = self.detector.detect(
                image=img,
                text_prompt="hand. object. tool. food. container.",
            )
            objects_for_match = [
                d for d in all_detections if "hand" not in d.label.lower()
            ]

            # Distinguish left/right hands
            hand_dicts = [
                {"label": h.label, "bbox": h.bbox, "confidence": h.confidence}
                for h in hands
            ]
            distinguished = distinguish_hands(hand_dicts)

            # Match hands to objects
            obj_dicts = [
                {"label": o.label, "bbox": o.bbox, "confidence": o.confidence}
                for o in objects_for_match
            ]
            match_result = match_hands_objects(distinguished, obj_dicts)
            builder.add_frame(match_result)

        result = builder.format_for_llm()
        if result:
            return f"[Hand-Object Dynamics]\n{result}"
        return ""
