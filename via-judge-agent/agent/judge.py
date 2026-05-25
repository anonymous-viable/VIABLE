"""
JudgeAgent -- main entry point for the VIA-Judge-Agent pipeline.

Workflow:
  Stage 0: Check I1/I2 (text-only, no visual evidence needed)
  Stage 1: Extract visual evidence (task-aware)
  Stage 2: Verify P/C/A failure types with evidence
  Stage 3: Merge all verdicts, select top-k
  Stage 4: Refinement -- let VLM pick the most critical top-k

Usage:
    from agent.judge import JudgeAgent
    agent = JudgeAgent()
    codes = agent.judge(frame_paths, question, candidate, task="walkvlm")
    # -> ["P1", "C3"] or ["N"]
"""

import json
import re
from typing import List

from agent.config.run_config import TOP_K, CONFIDENCE_THRESHOLD, DEVICE
from agent.config.failure_config import FAILURE_TAXONOMY, GROUP_P, GROUP_C, GROUP_A, GROUP_I
from agent.modules.vlm_backend import VLMBackend
from agent.modules.checklist_verifier import ChecklistVerifier
from agent.modules.evidence_extractor import VisualEvidenceExtractor


class JudgeAgent:
    """
    Main judge agent that orchestrates the full evaluation pipeline.
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
        Run the full judge pipeline.

        Args:
            frame_paths: paths to video frames (time-ordered)
            question: the user's question
            candidate: the candidate VLM answer to evaluate
            task: "walkvlm", "visassist", or "via_egodex"
            top_k: override max failure codes to return
            confidence_threshold: override min confidence

        Returns:
            List of failure codes, e.g. ["P1", "C3"], or ["N"] if no failures.
        """
        top_k = top_k or self.top_k
        confidence_threshold = confidence_threshold or self.confidence_threshold

        all_verdicts = []

        # Stage 0: I1/I2 check (text-only)
        i_verdicts = self._check_interaction(question, candidate)
        all_verdicts.extend(i_verdicts)

        # Stage 1: Extract visual evidence
        evidence = self.extractor.extract(images=frame_paths, task=task)

        # Stage 2: Verify P/C/A with evidence
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
        Stage 0: Check I1 (redundant) and I2 (truncated).
        These can be judged from text alone -- no visual evidence needed.
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
        Stage 2: Check P1-P4, C1-C3, A1-A3 with visual evidence.
        """
        all_verdicts = []

        # Group P: Perception (P1-P4)
        prompt_p = self.verifier.build_prompt(
            evidence=evidence,
            question=question,
            candidate=candidate,
            failure_codes=GROUP_P,
            group_name="Perception Fidelity",
        )
        raw_p = self.vlm.call(prompt_p, image_paths=frame_paths)
        all_verdicts.extend(self.verifier.parse_verdicts(raw_p))

        # Group C: Cognition (C1-C3)
        prompt_c = self.verifier.build_prompt(
            evidence=evidence,
            question=question,
            candidate=candidate,
            failure_codes=GROUP_C,
            group_name="Cognition Validity",
        )
        raw_c = self.vlm.call(prompt_c, image_paths=frame_paths)
        all_verdicts.extend(self.verifier.parse_verdicts(raw_c))

        # Group A: Action (A1-A3)
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

    def _get_fail_candidates(self, verdicts: List[dict], threshold: float = None) -> List[dict]:
        """Filter verdicts to only FAIL with confidence >= threshold."""
        threshold = threshold or self.confidence_threshold
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
        top_k: int = None,
    ) -> List[str]:
        """
        Stage 4: Refinement -- given multiple FAIL candidates, let VLM pick
        the most critical top-k considering severity and distinction rules.
        """
        top_k = top_k or self.top_k

        # Build candidate list for the prompt
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

        # Parse the output
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

        # Fallback: regex
        valid_codes = {
            "P1", "P2", "P3", "P4", "C1", "C2", "C3",
            "A1", "A2", "A3", "I1", "I2", "N",
        }
        found = re.findall(r"\b([PCAIN][1-4]?)\b", text)
        codes = [c for c in found if c in valid_codes]
        return codes[:top_k] if codes else ["N"]


if __name__ == "__main__":
    import argparse
    import os
    from agent.config.run_config import FRAME_ROOTS, DATA_ROOT

    parser = argparse.ArgumentParser(description="Run VIA-Judge-Agent on test data.")
    parser.add_argument("--task", required=True, choices=["visassist", "walkvlm", "via_egodex"])
    parser.add_argument("--split", default="single", choices=["single", "dual"])
    parser.add_argument("--output_dir", default="outputs/judge_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    from tqdm import tqdm
    from agent.modules.evidence_extractor import VisualEvidenceExtractor

    # Load data
    data_file = os.path.join(DATA_ROOT, f"{args.split}_injected_{args.task}_test.jsonl")
    with open(data_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if args.max_samples:
        samples = samples[:args.max_samples]

    # Output dir
    out_dir = os.path.join(args.output_dir, args.task, args.split)
    os.makedirs(out_dir, exist_ok=True)

    # Init agent
    agent = JudgeAgent()
    frame_root = FRAME_ROOTS.get(args.task, "")

    for sample in tqdm(samples, desc=f"{args.task}/{args.split}"):
        sample_id = sample["sample_id"]
        out_file = os.path.join(out_dir, f"{sample_id}.json")
        if os.path.exists(out_file):
            continue

        # Get frame paths
        frames_dir = os.path.join(frame_root, sample["frame_path"])
        if not os.path.isdir(frames_dir):
            continue
        exts = {".jpg", ".jpeg", ".png"}
        frame_paths = sorted(
            [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
             if os.path.splitext(f)[1].lower() in exts]
        )
        if not frame_paths:
            continue

        # Run judge
        try:
            codes = agent.judge(
                frame_paths=frame_paths,
                question=sample["question"],
                candidate=sample["modified_answer"],
                task=args.task,
            )
        except Exception as e:
            print(f"[Error] {sample_id}: {e}")
            continue

        # Save result
        result = {
            "sample_id": sample_id,
            "task": args.task,
            "split": args.split,
            "predicted": codes,
            "gt_labels": sample.get("failure_pair") or [sample.get("failure", "")],
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done. Results saved to {out_dir}/")
