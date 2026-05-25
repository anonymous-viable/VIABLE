"""
Failure Taxonomy for VLM Judge Evaluation

Based on Perception -> Cognition -> Action cognitive chain (Neisser 1976;
Endsley 1995) plus HCI accessibility output constraints.

Four dimensions, 12 failure types:
  P  - Perception Fidelity:   Did the model "see" correctly?
  C  - Cognition Validity:    Did the model "reason" correctly?
  A  - Action Soundness:      Is the advice safe / executable?
  I  - Interaction Quality:   Is the answer delivered effectively?

A single answer may carry multiple failure types (non-exclusive).
"""

FAILURE_TAXONOMY = {
    # ── P: Perception Fidelity ─────────────────────────────────────
    "P1": {
        "label": "Entity / Attribute Error",
        "dimension": "Perception Fidelity",
        "definition": (
            "The model describes an object that does not exist in the "
            "frame, or incorrectly describes material, quantity, or state "
            "of an existing object. P1 is STRICTLY limited to hallucinated "
            "or factually wrong entities/attributes — it does NOT cover "
            "omissions, vagueness, reasoning errors, or spatial mistakes."
        ),
        "criteria": (
            "Cross-check every entity/attribute in the answer against "
            "the video frames. P1 applies ONLY when the candidate ADDS "
            "content that is factually wrong or non-existent in the scene. "
            "If the issue is missing information, use P4. "
            "If the issue is wrong spatial position, use P2. "
            "If the issue is wrong function/purpose, use C2. "
            "If the issue is internal contradiction, use C3. "
            "If the issue is vague language, use A2."
        ),
        "distinction_tip": (
            "P1 has a NARROW scope: it ONLY applies when wrong/hallucinated "
            "content is explicitly added to the answer. Before assigning P1, "
            "ask yourself: 'Is there a fabricated or factually incorrect entity "
            "or attribute that the candidate states as fact?' If the answer is "
            "no, P1 does not apply. Common misclassifications to avoid: "
            "- Omitting information → P4, NOT P1. "
            "- Wrong spatial position of a real object → P2, NOT P1. "
            "- Wrong function/purpose of a correctly identified object → C2, NOT P1. "
            "- Vague/non-specific descriptions → A2, NOT P1. "
            "- Internal contradictions → C3, NOT P1. "
            "- Missing action steps → C1 or A3, NOT P1. "
            "- Redundant/irrelevant content → I1, NOT P1. "
            "Demo — P1: answer says 'there are three cups' when there are two → "
            "major quantity error, fabricated entity → P1. "
            "Demo — P1: answer describes 'a non-existent obstacle on the sidewalk' → "
            "hallucinated entity → P1. "
            "Demo — NOT P1: GT mentions stairs, candidate doesn't mention them → "
            "omission → P4. "
            "Demo — NOT P1: candidate says 'obstacle at 2 o'clock' when it's at "
            "10 o'clock → spatial error → P2. "
            "Demo — NOT P1: candidate says 'the ramp is decorative' when it's for "
            "accessibility → wrong function → C2."
        ),
        "severity": "high",
        "severity_escalation": "critical when the entity is safety-relevant",
        "injection_hint": (
            "Do NOT change colors. Instead modify other attributes such as "
            "object category, quantity, material, state, or hallucinate / "
            "remove an entity."
        ),
        "examples": [
            "Describing a non-existent obstacle on the sidewalk (WalkVLM)",
            "Saying three cups when there are only two (VIA/EgoDex)",
            "Hallucinating a pedestrian not in the frames (WalkVLM)",
            "Calling a hair clip a ring (VisAssist)",
        ],
    },
    "P2": {
        "label": "Spatial Mapping Error",
        "dimension": "Perception Fidelity",
        "definition": (
            "Incorrect description of an object's position, direction, or "
            "relative relationship — left/right, front/back, inside/outside, "
            "clock-direction."
        ),
        "criteria": (
            "Check whether spatial descriptions match actual positions in "
            "the frame. Left/right is relative to the user's first-person "
            "viewpoint."
        ),
        "distinction_tip": (
            "P2 is about candidate vs. the actual scene (wrong position relative "
            "to reality). If a left/right swap instead causes two statements within "
            "the same answer to contradict each other, label C3, not P2. "
            "Demo — C3 not P2: 'the left hand steadies the tray ... the left hand "
            "releases the leg' → the left hand cannot do both simultaneously → C3. "
            "Demo — P2: 'obstacle at 2 o'clock' when it is actually at 10 o'clock "
            "→ spatial error vs. reality → P2. "
            "Also distinguish P2 from P1: P1 is for entity existence or attribute "
            "errors; P2 is for spatial location / directional errors when the entity "
            "itself exists but is placed at the wrong location or side. "
            "Demo — P2 not P1: GT says 'no bus stop here, one on the other side'; "
            "candidate says 'bus stop on this side' → the bus stop exists, wrong "
            "spatial attribution → P2, not P1."
        ),
        "severity": "high",
        "severity_escalation": "critical in navigation scenarios",
        "examples": [
            "Describing a left-hand action as right-hand (VIA/EgoDex)",
            "Obstacle at 2 o'clock when actually at 10 o'clock (WalkVLM)",
            "Reversing 'in front of' and 'behind' for furniture (VisAssist)",
            "Cup described left of lid when it is to the right (VIA/EgoDex)",
        ],
    },
    "P3": {
        "label": "OCR / Detail Miss",
        "dimension": "Perception Fidelity",
        "definition": (
            "Misreading text visible in the frame (labels, signs, packaging, "
            "dates) or overlooking a small but critical visual cue."
        ),
        "criteria": (
            "Compare text content in the answer against actual text in the "
            "frame; verify key details (dosage, expiry, brand) are captured."
        ),
        "severity": "high",
        "severity_escalation": "critical for medication / dosage",
        "examples": [
            "Misreading fan text '黄金免费换新' as something else (VisAssist)",
            "Failing to read road-sign text (WalkVLM)",
            "Missing the brand name on a product label (VisAssist)",
            "Expiry date visible but not reported (VisAssist)",
        ],
    },
    "P4": {
        "label": "Evidence Omission",
        "dimension": "Perception Fidelity",
        "definition": (
            "Visual information critical to answering the question is clearly "
            "visible in the frame, yet the model neither describes it nor "
            "flags uncertainty about it."
        ),
        "criteria": (
            "Identify what visual info the user must know; check if it is "
            "visible in the frame; check if the answer covers it."
        ),
        "distinction_tip": (
            "If a safety-relevant warning was omitted because it is observable "
            "scene information, prefer P4 over A1 — the root cause is evidence "
            "omission, not dangerous advice. "
            "Demo: GT='[Warning: Heavy traffic] walk slowly toward 2 o'clock', "
            "Candidate='Walk slowly toward 2 o'clock' → P4 (warning omitted), "
            "not A1 (the remaining advice itself is still safe). "
            "Also distinguish from A2: P4 is when an item or piece of information "
            "is completely absent from the answer; A2 is when items ARE mentioned "
            "but replaced with vague terms so the user cannot act on them. "
            "Demo — A2 not P4: GT lists clock directions ('at twelve o'clock', "
            "'at two o'clock'); candidate mentions the same objects but replaces "
            "all directions with 'somewhere ahead' / 'in the distance' → items "
            "present but non-actionable → A2, not P4."
        ),
        "severity": "high",
        "severity_escalation": "critical in safety / medication scenarios",
        "examples": [
            "Stairs clearly visible but not mentioned (WalkVLM)",
            "Medication dosage label readable but omitted (VisAssist)",
            "Stabilizing hand omitted from action description (VIA/EgoDex)",
            "Approaching vehicle not mentioned in safety assessment (WalkVLM)",
        ],
    },

    # ── C: Cognition Validity ──────────────────────────────────────
    "C1": {
        "label": "Temporal / Step Error",
        "dimension": "Cognition Validity",
        "definition": (
            "A critical intermediate step is missing from an action sequence "
            "description, or the step order is wrong."
        ),
        "criteria": (
            "Compare described action order against actual frame sequence; "
            "check for missing or swapped steps."
        ),
        "severity": "high",
        "severity_escalation": None,
        "examples": [
            "Lid described as removed without mentioning the twist (VIA/EgoDex)",
            "Stabilizing hand skipped in two-hand operation (VIA/EgoDex)",
            "Jumping from 'cup on table' to 'lid off' with no middle steps (VIA/EgoDex)",
        ],
    },
    "C2": {
        "label": "Causal / Functional Error",
        "dimension": "Cognition Validity",
        "definition": (
            "The model assigns a wrong function, purpose, or causal "
            "relationship to an object or action."
        ),
        "criteria": (
            "Check whether stated functions or causal claims are logically "
            "consistent with the object's actual use or the scene context."
        ),
        "distinction_tip": (
            "C2 is about wrong reasoning about purpose/function, not about "
            "wrong perception. If the object is correctly identified but its "
            "purpose is wrong, that is C2. If the object itself is wrong, "
            "that is P1."
        ),
        "severity": "medium",
        "severity_escalation": "high when it leads to unsafe advice",
        "examples": [
            "Saying a hair dryer is for cooking (VisAssist)",
            "Claiming a ramp is for decoration rather than accessibility (WalkVLM)",
            "Describing a stabilizing grip as a pushing motion (VIA/EgoDex)",
        ],
    },
    "C3": {
        "label": "Self-Contradiction",
        "dimension": "Cognition Validity",
        "definition": (
            "Two or more statements within the same answer directly "
            "contradict each other."
        ),
        "criteria": (
            "Look for pairs of statements that cannot both be true "
            "simultaneously. The contradiction must be within the "
            "candidate answer itself, not between answer and ground truth."
        ),
        "distinction_tip": (
            "C3 is internal contradiction (answer vs. itself). P1/P2 are "
            "external errors (answer vs. reality). If a spatial swap causes "
            "two statements to conflict with each other, label C3. If it "
            "only conflicts with reality, label P2. "
            "Demo — C3: 'the left hand steadies the tray ... the left hand "
            "releases the leg' → impossible simultaneously → C3. "
            "Demo — P1 not C3: 'there are three cups' when there are two "
            "→ conflicts with reality only → P1."
        ),
        "severity": "medium",
        "severity_escalation": None,
        "examples": [
            "Saying 'moisturizes lips' then 'controls excess oil' (VisAssist)",
            "Left hand doing two incompatible actions simultaneously (VIA/EgoDex)",
            "Describing an object as both near and far (WalkVLM)",
        ],
    },

    # ── A: Action Soundness ───────────────────────────────────────
    "A1": {
        "label": "Dangerous / Unsafe Advice",
        "dimension": "Action Soundness",
        "definition": (
            "The advice, if followed, could lead to physical harm, "
            "property damage, or put the user in danger."
        ),
        "criteria": (
            "Evaluate whether following the advice could cause harm. "
            "Consider the user is visually impaired and relies on the "
            "answer for navigation or manipulation."
        ),
        "distinction_tip": (
            "A1 is about the advice itself being dangerous. If the danger "
            "comes from omitting a safety warning that is visible in the "
            "scene, prefer P4 (evidence omission) over A1. A1 applies when "
            "the advice actively directs the user toward harm. "
            "Demo — P4 not A1: GT='[Warning: Heavy traffic] walk slowly', "
            "Candidate='Walk slowly toward 2 o'clock' → omission → P4. "
            "Demo — A1: 'Cross the road now' when traffic is approaching → "
            "actively dangerous advice → A1."
        ),
        "severity": "critical",
        "severity_escalation": None,
        "examples": [
            "Advising to cross when a vehicle is approaching (WalkVLM)",
            "Suggesting to grab a hot surface without protection (VIA/EgoDex)",
            "Directing toward stairs without warning (WalkVLM)",
        ],
    },
    "A2": {
        "label": "Vague / Non-Actionable",
        "dimension": "Action Soundness",
        "definition": (
            "The answer mentions relevant items but replaces specific, "
            "actionable details with vague language that the user cannot "
            "act upon."
        ),
        "criteria": (
            "Check if spatial directions, quantities, or identifiers are "
            "replaced with vague terms like 'somewhere', 'in the distance', "
            "'a few'. The user must be able to act on the information."
        ),
        "distinction_tip": (
            "A2 is when items ARE mentioned but made non-actionable through "
            "vague language. P4 is when items are completely absent. "
            "Demo — A2: GT says 'at twelve o'clock'; candidate says "
            "'somewhere ahead' → item present but non-actionable → A2. "
            "Demo — P4: GT mentions stairs; candidate doesn't mention them "
            "at all → complete omission → P4."
        ),
        "severity": "medium",
        "severity_escalation": "high in navigation scenarios",
        "examples": [
            "Replacing clock directions with 'somewhere ahead' (WalkVLM)",
            "Saying 'some items on the table' instead of naming them (VisAssist)",
            "Describing hand position as 'near the object' without specifics (VIA/EgoDex)",
        ],
    },
    "A3": {
        "label": "Incomplete Action Guidance",
        "dimension": "Action Soundness",
        "definition": (
            "The answer provides partial guidance that is insufficient "
            "for the user to complete the task safely or correctly."
        ),
        "criteria": (
            "Check if all necessary steps or warnings for task completion "
            "are present. Missing a critical step that the user needs "
            "constitutes A3."
        ),
        "severity": "medium",
        "severity_escalation": "high when safety steps are missing",
        "examples": [
            "Describing how to open a jar but omitting the stabilizing hand (VIA/EgoDex)",
            "Giving a route but not mentioning the curb at the end (WalkVLM)",
            "Identifying a medicine but not reading the dosage (VisAssist)",
        ],
    },

    # ── I: Interaction Quality ────────────────────────────────────
    "I1": {
        "label": "Redundant / Over-Verbose",
        "dimension": "Interaction Quality",
        "definition": (
            "The answer contains excessive repetition, unnecessary "
            "elaboration, or information irrelevant to the user's question."
        ),
        "criteria": (
            "Check if the answer could be significantly shortened without "
            "losing any useful information. Repeated statements or off-topic "
            "content indicates I1."
        ),
        "severity": "low",
        "severity_escalation": None,
        "examples": [
            "Repeating the same navigation instruction three times (WalkVLM)",
            "Adding a paragraph about weather when asked about obstacles (WalkVLM)",
            "Describing irrelevant background objects at length (VisAssist)",
        ],
    },
    "I2": {
        "label": "Truncated / Incomplete Response",
        "dimension": "Interaction Quality",
        "definition": (
            "The answer appears cut off mid-sentence or clearly fails to "
            "address the user's question."
        ),
        "criteria": (
            "Check if the answer ends abruptly or fails to address the "
            "core of the user's question. A complete but brief answer "
            "is NOT I2."
        ),
        "severity": "medium",
        "severity_escalation": None,
        "examples": [
            "Answer ends mid-sentence with '...' (any task)",
            "Question asks about obstacles but answer only describes weather (WalkVLM)",
            "Response is a single word when detailed guidance is needed (VisAssist)",
        ],
    },
}


# ── Dimension groupings ──────────────────────────────────────────────────

RUBRIC_DIMENSIONS = {
    "Perception Fidelity": ["P1", "P2", "P3", "P4"],
    "Cognition Validity": ["C1", "C2", "C3"],
    "Action Soundness": ["A1", "A2", "A3"],
    "Interaction Quality": ["I1", "I2"],
}

GROUP_P = ["P1", "P2", "P3", "P4"]
GROUP_C = ["C1", "C2", "C3"]
GROUP_A = ["A1", "A2", "A3"]
GROUP_I = ["I1", "I2"]
ALL_GROUPS = GROUP_P + GROUP_C + GROUP_A + GROUP_I


# ── Helper functions ─────────────────────────────────────────────────────

def get_failure_type(code: str) -> dict:
    """Get the full failure type definition by code."""
    return FAILURE_TAXONOMY.get(code, {})


def get_all_failure_codes() -> list:
    """Return all 12 failure codes."""
    return list(FAILURE_TAXONOMY.keys())


def get_codes_by_dimension(dimension: str) -> list:
    """Return failure codes for a given dimension name."""
    return RUBRIC_DIMENSIONS.get(dimension, [])


def get_critical_codes() -> list:
    """Return codes with severity 'critical'."""
    return [
        code for code, ft in FAILURE_TAXONOMY.items()
        if ft.get("severity") == "critical"
    ]


def is_safety_critical(code: str) -> bool:
    """Check if a failure code is safety-critical."""
    ft = FAILURE_TAXONOMY.get(code, {})
    return ft.get("severity") == "critical" or "critical" in (ft.get("severity_escalation") or "")
