"""Checklist Verifier module.

Builds verification prompts for failure-type groups and parses VLM verdicts.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from agent.config.failure_config import FAILURE_TAXONOMY


class ChecklistVerifier:
    """Verifies candidate answers against failure-type checklists using a VLM backend."""

    def __init__(self, vlm_backend) -> None:
        """Initialize with a VLMBackend instance.

        Args:
            vlm_backend: A VLMBackend instance used for VLM inference.
        """
        self.vlm = vlm_backend

    def build_prompt(
        self,
        evidence: str,
        question: str,
        candidate: str,
        failure_codes: List[str],
        group_name: str,
    ) -> str:
        """Build a verification prompt for one group of failure types.

        Args:
            evidence: Visual evidence description from perception modules.
            question: The original VQA question.
            candidate: The candidate answer to evaluate.
            failure_codes: List of failure type codes (e.g. ["P1", "P2"]).
            group_name: Human-readable name for this failure group.

        Returns:
            Formatted prompt string for the VLM.
        """
        ft_block = ""
        for code in failure_codes:
            ft = FAILURE_TAXONOMY[code]
            ft_block += f"\n[{code}] {ft['label']}\n"
            ft_block += f"  Definition: {ft['definition']}\n"
            ft_block += f"  Criteria: {ft['criteria']}\n"
            if ft.get("distinction_tip"):
                ft_block += f"  Distinction: {ft['distinction_tip']}\n"

        return f"""\
You are a VQA judge for visually impaired assistance scenarios.
Your task: determine whether each failure type EXISTS in the candidate answer.

[Visual Evidence]:
{evidence}

[Question]: {question}

[Candidate Answer]: {candidate}

─── {group_name} Failure Types ───
{ft_block}
─── Instructions ───
For EACH failure type above, decide:
- PASS = this failure does NOT exist in the candidate answer
- FAIL = this failure EXISTS in the candidate answer

IMPORTANT constraints:
- The candidate answer contains AT MOST ONE type of failure from each dimension group.
  Do NOT assign multiple failure codes from the same group (e.g. do not assign both P1 and P2).
  If you suspect more than one, pick the single best-matching code with highest confidence.
- Be conservative: only mark FAIL when you have clear, concrete evidence.
- P1 (Entity/Attribute Error) has a NARROW scope: it ONLY applies when the candidate
  explicitly adds a fabricated or factually wrong entity/attribute. Do NOT use P1 for
  omissions (→P4), spatial errors (→P2), wrong function (→C2), vagueness (→A2),
  or contradictions (→C3). Ask: "Did the candidate fabricate something?" If no, P1 = PASS.

Output ONLY a JSON list. Each element: {{"code": "<code>", "verdict": "PASS"|"FAIL", "confidence": <0.0-1.0>}}
Example: [{{"code": "P1", "verdict": "FAIL", "confidence": 0.85}}, {{"code": "P2", "verdict": "PASS", "confidence": 0.9}}]
"""

    def parse_verdicts(self, raw_output: str) -> List[Dict]:
        """Parse VLM output into list of verdict dictionaries.

        Handles markdown code fences and attempts multiple JSON extraction
        strategies for robustness.

        Args:
            raw_output: Raw text output from the VLM.

        Returns:
            List of dicts with keys: code, verdict, confidence.
            Returns empty list if parsing fails.
        """
        text = raw_output.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # Attempt direct JSON parse
        try:
            results = json.loads(text)
            if isinstance(results, list):
                return results
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract first JSON array from text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass

        return []
