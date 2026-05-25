"""
Hand-object trajectory extraction module.

Merged from hand_tracking/contact_matcher.py, trajectory_builder.py,
and hand_trajectory_extractor.py into a single self-contained module.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from agent.modules.object_detector import DetectedObject


# ---------------------------------------------------------------------------
# Helper functions (from contact_matcher)
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def get_bbox_center(bbox):
    """
    Get center point of a bounding box.
    :param bbox: [x1, y1, x2, y2]
    :return: (center_x, center_y)
    """
    if bbox is None:
        return None
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return (int(round(center_x)), int(round(center_y)))


def distinguish_hands(hands):
    """
    Distinguish left and right hands based on position.
    Left side of image = left hand, right side = right hand.
    """
    # Filter out hands with None bbox
    hands = [h for h in hands if h.get('bbox') is not None]
    if len(hands) == 0:
        return None, None
    if len(hands) == 1:
        hand = hands[0]
        return hand, None
    hands_sorted = sorted(hands, key=lambda h: (h['bbox'][0] + h['bbox'][2]) / 2)
    left_hand = hands_sorted[0]
    right_hand = hands_sorted[-1]
    return left_hand, right_hand


def match_hand_to_object(hand, objects, iou_threshold=0.01):
    """
    Find the contacted object for a hand (the one with maximum IoU).
    :param hand: {'bbox': [x1,y1,x2,y2], ...}
    :param objects: [{'bbox': [x1,y1,x2,y2], 'name': 'cup', ...}, ...]
    :param iou_threshold: IoU threshold, contact is determined if exceeded
    :return: Contacted object dict, or None
    """
    if hand is None or len(objects) == 0:
        return None

    max_iou = 0
    matched_obj = None

    for obj in objects:
        iou = compute_iou(hand['bbox'], obj['bbox'])
        if iou > max_iou and iou > iou_threshold:
            max_iou = iou
            matched_obj = obj

    return matched_obj


def find_two_hand_object(left_hand, right_hand, objects, iou_threshold=0.01):
    """
    Find object that is contacted by both hands.
    :param left_hand: Left hand detection result
    :param right_hand: Right hand detection result
    :param objects: List of detected objects
    :param iou_threshold: IoU threshold
    :return: Object contacted by both hands, or None
    """
    if left_hand is None or right_hand is None or len(objects) == 0:
        return None

    for obj in objects:
        left_iou = compute_iou(left_hand['bbox'], obj['bbox'])
        right_iou = compute_iou(right_hand['bbox'], obj['bbox'])

        # Both hands have contact with this object
        if left_iou > iou_threshold and right_iou > iou_threshold:
            return obj

    return None


def match_hands_objects(hands, objects, iou_threshold=0.01):
    """
    Complete matching: distinguish left/right hands + find contacted objects for each.

    :param hands: List of hand detection results
    :param objects: List of object detection results
    :param iou_threshold: IoU threshold
    :return: {
        'left_hand': {...} or None,
        'right_hand': {...} or None,
        'left_object': {...} or None,
        'right_object': {...} or None,
        'two_hand_object': {...} or None
    }
    """
    # Distinguish left and right hands
    left_hand, right_hand = distinguish_hands(hands)

    # Match objects to each hand
    left_object = match_hand_to_object(left_hand, objects, iou_threshold)
    right_object = match_hand_to_object(right_hand, objects, iou_threshold)

    # Find object contacted by both hands
    two_hand_object = find_two_hand_object(left_hand, right_hand, objects, iou_threshold)

    return {
        'left_hand': left_hand,
        'right_hand': right_hand,
        'left_object': left_object,
        'right_object': right_object,
        'two_hand_object': two_hand_object
    }


# ---------------------------------------------------------------------------
# TrajectoryBuilder class (from trajectory_builder)
# ---------------------------------------------------------------------------

class TrajectoryBuilder:
    """Build trajectories from frame-by-frame detections."""

    def __init__(self, use_relative_direction=False):
        self.use_relative_direction = use_relative_direction

        self.left_hand_trajectory = []
        self.right_hand_trajectory = []
        self.left_object_trajectory = []
        self.right_object_trajectory = []
        self.two_hand_object_trajectory = []

        # Store object names
        self.left_object_names = []
        self.right_object_names = []
        self.two_hand_object_names = []

    def add_frame(self, match_result):
        """
        Add a frame's detection result to trajectories.
        :param match_result: Output from match_hands_objects()
        """
        # Extract bounding boxes
        left_hand = match_result.get('left_hand')
        right_hand = match_result.get('right_hand')
        left_object = match_result.get('left_object')
        right_object = match_result.get('right_object')
        two_hand_object = match_result.get('two_hand_object')

        # Calculate center points and add to trajectories
        self.left_hand_trajectory.append(
            get_bbox_center(left_hand['bbox']) if left_hand else None
        )
        self.right_hand_trajectory.append(
            get_bbox_center(right_hand['bbox']) if right_hand else None
        )

        # For objects, store both center point and name
        self.left_object_trajectory.append(
            get_bbox_center(left_object['bbox']) if left_object else None
        )
        self.left_object_names.append(
            left_object.get('name', 'unknown') if left_object else None
        )

        self.right_object_trajectory.append(
            get_bbox_center(right_object['bbox']) if right_object else None
        )
        self.right_object_names.append(
            right_object.get('name', 'unknown') if right_object else None
        )

        self.two_hand_object_trajectory.append(
            get_bbox_center(two_hand_object['bbox']) if two_hand_object else None
        )
        self.two_hand_object_names.append(
            two_hand_object.get('name', 'unknown') if two_hand_object else None
        )

    def get_trajectories(self):
        """Get all trajectories."""
        return {
            'left_hand': self.left_hand_trajectory,
            'right_hand': self.right_hand_trajectory,
            'left_object': self.left_object_trajectory,
            'right_object': self.right_object_trajectory,
            'two_hand_object': self.two_hand_object_trajectory
        }

    def calculate_direction(self, p1, p2, threshold=10):
        """
        Calculate direction from p1 to p2.
        :param p1: (x1, y1)
        :param p2: (x2, y2)
        :param threshold: Minimum distance to consider as movement
        :return: Direction string like "right", "up-left", "still"
        """
        if p1 is None or p2 is None:
            return "none"

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        distance = math.sqrt(dx**2 + dy**2)

        # Too small to consider movement
        if distance < threshold:
            return "still"

        # Calculate 8 directions
        angle = math.atan2(dy, dx)  # Note: y-axis is downward in images
        angle_deg = math.degrees(angle)

        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360

        # 8 directions (45 degrees each)
        if angle_deg < 22.5 or angle_deg >= 337.5:
            return "right"
        elif angle_deg < 67.5:
            return "down-right"
        elif angle_deg < 112.5:
            return "down"
        elif angle_deg < 157.5:
            return "down-left"
        elif angle_deg < 202.5:
            return "left"
        elif angle_deg < 247.5:
            return "up-left"
        elif angle_deg < 292.5:
            return "up"
        else:
            return "up-right"

    def trajectory_to_directions(self, trajectory):
        """
        Convert trajectory to direction sequence.
        :param trajectory: List of (x, y) points
        :return: List of directions
        """
        if len(trajectory) < 2:
            return []

        directions = []
        for i in range(len(trajectory) - 1):
            direction = self.calculate_direction(trajectory[i], trajectory[i + 1])
            directions.append(direction)

        return directions

    def format_trajectory(self, trajectory, names=None):
        """
        Format trajectory as string.
        If use_relative_direction=True: first frame as (x,y), then directions
        If use_relative_direction=False: all frames as (x,y)
        """
        if len(trajectory) == 0:
            return "()"

        if self.use_relative_direction:
            result = []
            first_point = trajectory[0]
            if first_point is None:
                if names and names[0]:
                    result.append(f"(None,None,{names[0]})")
                else:
                    result.append("(None,None)")
            else:
                if names and names[0]:
                    result.append(f"({first_point[0]},{first_point[1]},{names[0]})")
                else:
                    result.append(f"({first_point[0]},{first_point[1]})")

            directions = self.trajectory_to_directions(trajectory)
            for i, direction in enumerate(directions):
                if names and i + 1 < len(names) and names[i + 1]:
                    result.append(f"{direction},{names[i + 1]}")
                else:
                    result.append(direction)

            return "(" + ",".join(result) + ")"
        else:
            points = []
            for i, point in enumerate(trajectory):
                if point is None:
                    if names and i < len(names) and names[i]:
                        points.append(f"(None,None,{names[i]})")
                    else:
                        points.append("(None,None)")
                else:
                    if names and i < len(names) and names[i]:
                        points.append(f"({point[0]},{point[1]},{names[i]})")
                    else:
                        points.append(f"({point[0]},{point[1]})")
            return "(" + ",".join(points) + ")"

    def get_object_summary(self, object_names):
        """Summarize object changes across frames."""
        if not object_names or all(name is None for name in object_names):
            return "none"

        valid_names = [name for name in object_names if name is not None]
        if not valid_names:
            return "none"

        unique_names = list(set(valid_names))
        if len(unique_names) == 1:
            return unique_names[0]

        return " -> ".join(unique_names)

    def format_for_llm(self):
        """Format trajectories in a human-readable way for LLM prompts."""
        output = ""

        if self.left_hand_trajectory:
            directions = self.trajectory_to_directions(self.left_hand_trajectory)
            direction_str = " -> ".join(directions) if directions else "stationary"
            obj_summary = self.get_object_summary(self.left_object_names)
            output += f"Left hand: {direction_str} (object: {obj_summary})\n"

        if self.right_hand_trajectory:
            directions = self.trajectory_to_directions(self.right_hand_trajectory)
            direction_str = " -> ".join(directions) if directions else "stationary"
            obj_summary = self.get_object_summary(self.right_object_names)
            output += f"Right hand: {direction_str} (object: {obj_summary})\n"

        two_hand_summary = self.get_object_summary(self.two_hand_object_names)
        if two_hand_summary != "none":
            output += f"Two-hand object: {two_hand_summary}\n"

        return output.strip()

    def format_all_trajectories(self):
        """Format all trajectories as the required output format."""
        output = "## Hand Object Dynamics\n"
        output += f"left hand:{self.format_trajectory(self.left_hand_trajectory)}\n"
        output += f"right hand:{self.format_trajectory(self.right_hand_trajectory)}\n"
        output += f"left hand object:{self.format_trajectory(self.left_object_trajectory, self.left_object_names)}\n"
        output += f"right hand object:{self.format_trajectory(self.right_object_trajectory, self.right_object_names)}\n"
        output += f"two hand object:{self.format_trajectory(self.two_hand_object_trajectory, self.two_hand_object_names)}\n"
        return output
