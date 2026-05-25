"""
Ablation variant: Workflow Only

Structured multi-stage workflow (grouped verification + distinction tips +
refinement) but NO CV tool evidence. Tests the independent contribution
of the workflow design.

Usage:
    from agent.workflow_only import WorkflowOnlyJudge
    judge = WorkflowOnlyJudge()
    codes = judge.judge(frame_paths, question, candidate, task="walkvlm")
"""

import json
import re
from typing import List

from agent.config.run_config import TOP_K, CONFIDENCE_THRESHOLD
from agent.config.failure_config import (
    FAILURE_TAXONOMY, GROUP_P, GROUP_C, GROUP_A, GROUP_I,
)
from agent.modules.vlm_backend import VLMBackend
from agent.modules.checklist_verifier import ChecklistVerifier


class WorkflowOnlyJudge:
    """
    Ablation: Structured workflow without tool augmentation.
    Same multi-stage pipeline as the full agent (Stage 0 → 2 → 3 → 4),
    but evidence is always empty.
    """

    def __init__(
        self,
        top_k: int = None,
        confidence_threshold: float = None,
    ):
        self.top_k = top_k or TOP_K
        self.confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD

        self.vlm = VLMBackend()
        self.verifier = ChecklistVerifier(self.vlm)

    def judge(
        self,
        frame_paths: List[str],
        question: str,
        candidate: str,
        task: str = "",
        top_k: int = None,
        confidence_threshold: float = None,
    ) -> List[str]:
        """
        Run workflow-only judge: structured stages, no CV evidence.

        Same signature as JudgeAgent.judge() for drop-in compatibility.
        """
        top_k = top_k or self.top_k
        confidence_threshold = confidence_threshold or self.confidence_threshold

        evidence = "[No tool-extracted evidence available]"

        all_verdicts = []

        # Stage 0: I1/I2 (text-only)
        i_verdicts = self._check_interaction(question, candidate)
        all_verdicts.extend(i_verdicts)

        # Stage 2: P/C/A with structured workflow (no evidence)
        pca_verdicts = self._check_pca(evidence, question, candidate, frame_paths)
        all_verdicts.extend(pca_verdicts)

        # Stage 3: Filter candidates
        candidates_fail = self._get_fail_candidates(all_verdicts, confidence_threshold)

        # Stage 4: Refinement
        if len(candidates_fail) <= top_k:
            return [v["code"] for v in candidates_fail] if candidates_fail else ["N"]

        refined = self._refine_selection(
            candidates_fail, evidence, question, candidate, frame_paths, top_k
        )
        return refined if refined else ["N"]

    def _check_interaction(self, question: str, candidate: str) -> List[dict]:
        """
        Stage 0: Check I1/I2 (text-only).
        Identical to JudgeAgent._check_interaction.
        """
        codes = GROUP_I
        ft_block = ""
        for code in codes:
            ft = FAILURE_TAXONOMY[code]
            ft_block += f"\n[{code}] {ft['label']}\n"
            ft_block += f"  Definition: {ft['definition']}\n"
            ft_block += f"  Criteria: {ft['criteria']}\n"

        prompt = f"""\
You are a VQA judge for visually impaired assistance scenarios.
Check ONLY interaction quality issues (no visual evidence needed).

[Question]: {question}

[Candidate Answer]: {candidate}

--- Interaction Quality ---
{ft_block}
--- Instructions ---
For EACH failure type above, decide:
- PASS = this failure does NOT exist
- FAIL = this failure EXISTS

IMPORTANT: I1 (Redundancy) and I2 (Truncation) should ONLY be marked as FAIL
when you are highly confident (confidence >= 0.8). These are subtle issues —
only flag them when the evidence is unambiguous:
- I1: The answer contains clearly irrelevant or repeated information that adds no value.
- I2: The answer is obviously cut off mid-sentence or missing critical information.
If in doubt, mark as PASS.

Output ONLY a JSON list: [{{"code": "I1", "verdict": "PASS"|"FAIL", "confidence": 0.0-1.0}}, ...]
"""
        raw = self.vlm.call(prompt, image_paths=None)
        return self.verifier.parse_verdicts(raw)

    def _check_pca(
        self,
        evidence: str,
        question: str,
        candidate: str,
        frame_paths: List[str],
    ) -> List[dict]:
        """
        Stage 2: Check P/C/A with grouped verification + distinction tips.
        Identical to JudgeAgent._check_pca.
        """
        all_verdicts = []

        prompt_p = self.verifier.build_prompt(
            evidence=evidence,
            question=question,
            candidate=candidate,
            failure_codes=GROUP_P,
            group_name="Perception Fidelity",
        )
        raw_p = self.vlm.call(prompt_p, image_paths=frame_paths)
        all_verdicts.extend(self.verifier.parse_verdicts(raw_p))

        prompt_c = self.verifier.build_prompt(
            evidence=evidence,
            question=question,
            candidate=candidate,
            failure_codes=GROUP_C,
            group_name="Cognition Validity",
        )
        raw_c = self.vlm.call(prompt_c, image_paths=frame_paths)
        all_verdicts.extend(self.verifier.parse_verdicts(raw_c))

        prompt_a = self.verifier.build_prompt(
            evidence=evidence,
            question=question,
            candidate=candidate,
            failure_codes=GROUP_A,
            group_name="Action Soundness",
        )
        raw_a = self.vlm.call(prompt_a, image_paths=frame_paths)
        all_verdicts.extend(self.verifier.parse_verdicts(raw_a))

        return all_verdicts

    def _get_fail_candidates(self, verdicts: List[dict], threshold: float) -> List[dict]:
        """Filter verdicts to only FAIL with confidence >= threshold."""
        fails = [
            v for v in verdicts
            if v.get("verdict", "").upper() == "FAIL"
            and v.get("confidence", 0) >= threshold
        ]
        fails.sort(key=lambda v: v.get("confidence", 0), reverse=True)
        return fails

    def _refine_selection(
        self,
        candidates: List[dict],
        evidence: str,
        question: str,
        candidate: str,
        frame_paths: List[str],
        top_k: int,
    ) -> List[str]:
        """
        Stage 4: Refinement.
        Identical to JudgeAgent._refine_selection.
        """
        candidate_block = ""
        for v in candidates:
            code = v["code"]
            ft = FAILURE_TAXONOMY[code]
            candidate_block += (
                f"  - [{code}] {ft['label']} (confidence={v.get('confidence', '?')})\n"
            )

        prompt = f"""\
You are a VQA judge for visually impaired assistance scenarios.

The following failures were detected in the candidate answer.
Your task: select the {top_k} MOST CRITICAL failures, ranked by severity.

[Visual Evidence]:
{evidence}

[Question]: {question}

[Candidate Answer]: {candidate}

--- Detected Failures ---
{candidate_block}
--- Selection Rules ---
- Pick at most {top_k} failures that are MOST severe and MOST clearly present.
- The candidate answer contains AT MOST ONE failure per dimension group (P/C/A/I).
  Do NOT select two codes from the same group.
- If two failures overlap, pick the one
  that better matches the root cause based on the distinction rules.
- Safety-critical failures (A1) take priority.
- I1/I2 should only be selected if you are highly confident they exist.
- If no failure is clearly present after reconsideration, output ["N"].

Output ONLY a JSON list of failure codes, e.g. ["P4", "C3"] or ["A1"] or ["N"].
No explanation, no extra text.
"""
        raw = self.vlm.call(prompt, image_paths=frame_paths)

        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(c) for c in result[:top_k]]
        except (json.JSONDecodeError, ValueError):
            pass

        valid_codes = {
            "P1", "P2", "P3", "P4", "C1", "C2", "C3",
            "A1", "A2", "A3", "I1", "I2", "N",
        }
        found = re.findall(r"\b([PCAIN][1-4]?)\b", text)
        codes = [c for c in found if c in valid_codes]
        return codes[:top_k] if codes else ["N"]
