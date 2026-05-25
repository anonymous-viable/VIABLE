"""
Ablation variant: Tool Only

CV evidence extraction (full perception pipeline) but NO structured workflow.
Single VLM call with all 12 failure types at once — no distinction tips,
no grouped verification stages, no refinement.

Usage:
    from agent.tool_only import ToolOnlyJudge
    judge = ToolOnlyJudge()
    codes = judge.judge(frame_paths, question, candidate, task="walkvlm")
"""

from typing import List

from agent.config.run_config import TOP_K, CONFIDENCE_THRESHOLD, DEVICE
from agent.config.failure_config import FAILURE_TAXONOMY
from agent.modules.vlm_backend import VLMBackend
from agent.modules.checklist_verifier import ChecklistVerifier
from agent.modules.evidence_extractor import VisualEvidenceExtractor


class ToolOnlyJudge:
    """
    Ablation: Tool augmentation without structured workflow.
    Extracts CV evidence (same as full agent Stage 1), then makes a single
    VLM call with all 12 failure types — no distinction tips, no grouped
    verification, no refinement.
    """

    def __init__(
        self,
        top_k: int = None,
        confidence_threshold: float = None,
        device: str = None,
    ):
        self.top_k = top_k or TOP_K
        self.confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
        self.device = device or DEVICE

        self.vlm = VLMBackend()
        self.verifier = ChecklistVerifier(self.vlm)
        self._extractor = None  # lazy init

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = VisualEvidenceExtractor(device=self.device)
        return self._extractor

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
        Run tool-only judge: extract evidence, single VLM call.

        Same signature as JudgeAgent.judge() for drop-in compatibility.
        """
        top_k = top_k or self.top_k
        confidence_threshold = confidence_threshold or self.confidence_threshold

        # Stage 1: Extract visual evidence (same as full agent)
        evidence = self.extractor.extract(images=frame_paths, task=task)

        # Single VLM call: all 12 types, only definition (no criteria/distinction)
        ft_block = ""
        all_codes = [
            "P1", "P2", "P3", "P4", "C1", "C2", "C3",
            "A1", "A2", "A3", "I1", "I2",
        ]
        for code in all_codes:
            ft = FAILURE_TAXONOMY[code]
            ft_block += f"\n[{code}] {ft['label']}\n"
            ft_block += f"  Definition: {ft['definition']}\n"

        prompt = f"""\
You are a VQA judge for visually impaired assistance scenarios.
Evaluate the candidate answer using the visual evidence below.

[Visual Evidence]:
{evidence}

[Question]: {question}

[Candidate Answer]: {candidate}

--- Failure Types ---
{ft_block}
--- Instructions ---
For EACH failure type, decide PASS or FAIL.
Output ONLY a JSON list: [{{"code": "P1", "verdict": "PASS"|"FAIL", "confidence": 0.0-1.0}}, ...]
"""
        raw = self.vlm.call(prompt, image_paths=frame_paths)
        verdicts = self.verifier.parse_verdicts(raw)

        # Filter and select top-k
        fails = [
            v for v in verdicts
            if v.get("verdict", "").upper() == "FAIL"
            and v.get("confidence", 0) >= confidence_threshold
        ]
        fails.sort(key=lambda v: v.get("confidence", 0), reverse=True)

        if not fails:
            return ["N"]
        return [v["code"] for v in fails[:top_k]]
