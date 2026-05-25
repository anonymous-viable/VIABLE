"""Object Detection module — Grounded-SAM-2 (open-vocabulary detection + segmentation)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(src: Union[str, Image.Image]) -> Image.Image:
    """Unified loader: path or PIL.Image -> PIL.Image (RGB)."""
    if isinstance(src, Image.Image):
        return src
    return Image.open(src).convert("RGB")


def _load_images(sources: List[Union[str, Image.Image]]) -> List[Image.Image]:
    return [_load_image(s) for s in sources]


def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    mask_area_ratio: Optional[float] = None  # mask area as fraction of image
    centroid: Optional[tuple] = None  # (cx, cy)

    def to_text(self) -> str:
        parts = [
            f"{self.label} (conf={self.confidence:.2f})",
            f"bbox=({self.bbox[0]:.0f},{self.bbox[1]:.0f},{self.bbox[2]:.0f},{self.bbox[3]:.0f})",
        ]
        if self.centroid is not None:
            parts.append(f"center=({self.centroid[0]:.0f},{self.centroid[1]:.0f})")
        if self.mask_area_ratio is not None:
            parts.append(f"area={self.mask_area_ratio:.1%}")
        return ", ".join(parts)


# =========================================================================
#  ObjectDetector — Grounded-SAM-2
# =========================================================================

class ObjectDetector:
    """
    Grounded-SAM-2: GroundingDINO (text->bbox) + SAM2 (bbox->mask).

    Dependencies:
        pip install grounding-dino-py segment-anything-2
    """

    def __init__(
        self,
        grounding_model_id: str = None,
        sam2_checkpoint: str = None,
        device: str = "cuda:0",
    ):
        from agent.config import run_config as _cfg
        self.device = device
        self._grounding_model_id = grounding_model_id or _cfg.GROUNDING_DINO_PATH
        self._sam2_checkpoint = sam2_checkpoint or _cfg.SAM2_CHECKPOINT
        # lazy init
        self._grounding_processor = None
        self._grounding_model = None
        self._sam2_predictor = None

    # ---- lazy init ----
    def _ensure_loaded(self):
        if self._grounding_model is not None:
            return

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self._grounding_processor = AutoProcessor.from_pretrained(
            self._grounding_model_id
        )
        self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._grounding_model_id
        ).to(self.device)
        self._grounding_model.eval()

        # SAM2 — load from local checkpoint
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(
            config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
            ckpt_path=os.path.join(self._sam2_checkpoint, "sam2.1_hiera_large.pt"),
            device=self.device,
        )
        self._sam2_predictor = SAM2ImagePredictor(sam2_model)

    # ---- core ----
    def detect(
        self,
        image: Union[str, Image.Image],
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> List[DetectedObject]:
        """
        Run open-vocabulary detection + segmentation on a single frame.

        Args:
            image: image path or PIL.Image
            text_prompt: detection text, e.g. "cat. dog. table."
            box_threshold: detection box confidence threshold
            text_threshold: text matching threshold

        Returns:
            List[DetectedObject]
        """
        self._ensure_loaded()
        pil_img = _load_image(image)
        w, h = pil_img.size

        # --- GroundingDINO detection ---
        inputs = self._grounding_processor(
            images=pil_img, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._grounding_model(**inputs)

        results = self._grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"].cpu().numpy()    # (N, 4) xyxy
        scores = results["scores"].cpu().numpy()  # (N,)
        labels = results["labels"]                # list[str]

        if len(boxes) == 0:
            return []

        # --- SAM2 segmentation ---
        self._sam2_predictor.set_image(np.array(pil_img))
        masks_list = []
        for box in boxes:
            masks, _, _ = self._sam2_predictor.predict(
                box=box, multimask_output=False
            )
            masks_list.append(masks[0])  # (H, W) bool

        # --- assemble results ---
        total_pixels = h * w
        detected = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            mask = masks_list[i] if i < len(masks_list) else None
            mask_ratio = float(mask.sum()) / total_pixels if mask is not None else None
            cy, cx = None, None
            if mask is not None:
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx, cy = float(xs.mean()), float(ys.mean())
            detected.append(DetectedObject(
                label=label.strip(),
                confidence=float(score),
                bbox=tuple(box.tolist()),
                mask_area_ratio=mask_ratio,
                centroid=(cx, cy) if cx is not None else None,
            ))
        return detected

    def detect_to_text(
        self,
        image: Union[str, Image.Image],
        text_prompt: str,
        **kwargs,
    ) -> str:
        """Detect and return a text description directly."""
        objects = self.detect(image, text_prompt, **kwargs)
        if not objects:
            return "[Objects] No objects detected."
        lines = ["[Objects]"] + [f"  - {obj.to_text()}" for obj in objects]
        return "\n".join(lines)
