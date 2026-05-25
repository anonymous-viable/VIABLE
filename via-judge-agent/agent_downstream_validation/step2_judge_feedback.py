"""
Downstream Validation — Step 2: Judge sampled predictions.

Reads sampled predictions from step1 output, runs two judge conditions:
  1. Raw judge: single VLM call, no evidence, no workflow
  2. Harnessed judge (VIA-Judge): full pipeline with CV evidence + structured verification

Usage:
    python step2_judge_feedback.py --task visassist --model qwen_instruction_8B --mode harnessed
    python step2_judge_feedback.py --task all --model all --mode raw
    python step2_judge_feedback.py --task walkvlm --model gpt_5_4 --mode both
"""

import json
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import run_config as config
from agent.config.failure_config import FAILURE_TAXONOMY
from agent.judge import JudgeAgent
from agent.modules.vlm_backend import VLMBackend
from agent.modules.evidence_extractor import VisualEvidenceExtractor

# ── Config ────────────────────────────────────────────────────────────────────
STEP1_DIR = "outputs/step1_predictions"
OUTPUT_DIR = "outputs/step2_judge_feedback"

TASKS = ["visassist", "walkvlm", "via_egodex"]
MODELS = ["qwen_instruction_8B", "gpt_5_4"]

FRAME_ROOTS = {
    "visassist":  "PATH/TO/VisAssist/frames",
    "walkvlm":    "PATH/TO/WalkVLM/frames",
    "via_egodex": "PATH/TO/EgoDex/frames",
}


def get_frame_paths(frame_path_field: str, task: str) -> list:
    """Get sorted image paths from frame_path field."""
    frame_root = FRAME_ROOTS.get(task, "")
    frames_dir = os.path.join(frame_root, frame_path_field)
    if not os.path.isdir(frames_dir):
        return []
    exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in os.listdir(frames_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    return [os.path.join(frames_dir, f) for f in files]


# ── Raw Judge (no evidence, no workflow) ─────────────────────────────────────

def generate_raw_feedback(vlm: VLMBackend, question: str, prediction: str, frame_paths: list) -> dict:
    """Single VLM call — just ask if there are problems. No ground truth."""
    prompt = f"""\
You are evaluating a VLM response given to a visually impaired user.
Judge the response for errors based on what you can see in the images.

[Question]: {question}

[VLM Response]: {prediction}

Identify any problems: incorrect information, hallucinations, spatial errors,
missing critical details, safety issues, redundancy, etc.

Output ONLY a JSON object:
{{"failure_codes": ["P1", "C3"], "feedback": "concise actionable feedback"}}
Or if no failures:
{{"failure_codes": ["N"], "feedback": "response is adequate"}}
"""
    raw = vlm.call(prompt, image_paths=frame_paths[:4])
    return _parse_feedback(raw)


# ── Harnessed Judge (full VIA-Judge pipeline) ────────────────────────────────

def generate_harnessed_feedback(
    judge: JudgeAgent, vlm: VLMBackend,
    question: str, prediction: str, frame_paths: list, task: str
) -> dict:
    """Full pipeline: CV evidence + structured verification. No ground truth."""
    # Extract visual evidence
    evidence = judge.extractor.extract(images=frame_paths, task=task)

    # Run structured checks
    i_verdicts = judge._check_interaction(question, prediction)
    pca_verdicts = judge._check_pca(evidence, question, prediction, frame_paths)
    all_verdicts = i_verdicts + pca_verdicts

    # Filter failures
    fails = judge._get_fail_candidates(all_verdicts, config.CONFIDENCE_THRESHOLD)
    if not fails:
        return {"failure_codes": ["N"], "feedback": "No significant issues found."}

    top_fails = fails[:3]
    failure_codes = [v["code"] for v in top_fails]

    # Generate feedback based on diagnosed failures
    fail_descriptions = ""
    for v in top_fails:
        code = v["code"]
        ft = FAILURE_TAXONOMY[code]
        fail_descriptions += f"- [{code}] {ft['label']}: {ft['definition']}\n"

    feedback_prompt = f"""\
You are providing feedback to improve a VLM response for a visually impaired user.

[Question]: {question}

[VLM Response]: {prediction}

[Visual Evidence]:
{evidence}

[Diagnosed Failures]:
{fail_descriptions}

Write concise, actionable feedback (2-4 sentences) telling the responder
exactly what to fix. Be specific.

Output ONLY the feedback text.
"""
    feedback_text = vlm.call(feedback_prompt, image_paths=frame_paths[:4])
    return {"failure_codes": failure_codes, "feedback": feedback_text.strip()}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_feedback(raw: str) -> dict:
    """Parse VLM output into {failure_codes, feedback}."""
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return {
                "failure_codes": result.get("failure_codes", []),
                "feedback": result.get("feedback", text),
            }
    except (json.JSONDecodeError, ValueError):
        pass
    return {"failure_codes": [], "feedback": raw.strip()}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 2: Judge sampled predictions")
    parser.add_argument("--task", default="all", choices=TASKS + ["all"])
    parser.add_argument("--model", default="all", choices=MODELS + ["all"])
    parser.add_argument("--mode", default="both", choices=["raw", "harnessed", "both"])
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    models = MODELS if args.model == "all" else [args.model]

    # Determine judge subdirectory based on backend
    if config.BACKEND == "openai":
        judge_subdir = "judge_gpt"
    else:
        judge_subdir = "judge_qwen"

    # Init
    vlm = VLMBackend()
    judge = JudgeAgent() if args.mode in ("harnessed", "both") else None

    base_dir = Path(__file__).parent
    out_dir = base_dir / args.output_dir / judge_subdir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Backend: {config.BACKEND}, Mode: {args.mode}, Output: {out_dir}")

    for task in tasks:
        for model in models:
            input_file = base_dir / STEP1_DIR / f"{task}_{model}_sampled.jsonl"
            if not input_file.exists():
                print(f"[Skip] {input_file.name} not found")
                continue

            # Load sampled predictions
            with open(input_file, encoding="utf-8") as f:
                items = [json.loads(line) for line in f if line.strip()]

            print(f"\n[{task}/{model}] {len(items)} sampled predictions")

            # Output file
            output_file = out_dir / f"{task}_{model}_{args.mode}.jsonl"

            # Load existing results for resume
            done_indices = set()
            if output_file.exists():
                with open(output_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                r = json.loads(line)
                                done_indices.add(r.get("index"))
                            except json.JSONDecodeError:
                                pass
                print(f"  Resuming: {len(done_indices)} already done")

            with open(output_file, "a", encoding="utf-8") as fout:
                for item in tqdm(items, desc=f"{task}/{model} ({args.mode})"):
                    if item.get("index") in done_indices:
                        continue

                    prediction = item.get("prediction", "")
                    if not prediction:
                        continue

                    question = item.get("question", "")
                    item_task = item.get("task", task)
                    frame_path = item.get("frame_path", "")
                    frame_paths = get_frame_paths(frame_path, item_task)

                    result = {**item}

                    try:
                        if args.mode in ("raw", "both"):
                            raw_fb = generate_raw_feedback(vlm, question, prediction, frame_paths)
                            result["raw_judge"] = raw_fb

                        if args.mode in ("harnessed", "both"):
                            harnessed_fb = generate_harnessed_feedback(
                                judge, vlm, question, prediction, frame_paths, item_task
                            )
                            result["harnessed_judge"] = harnessed_fb
                    except Exception as e:
                        print(f"\n  [Error] index={item.get('index')}: {e}")
                        import time
                        time.sleep(5)
                        continue

                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()

            print(f"  Saved: {output_file.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
