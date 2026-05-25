import sys
import os
import json
from tqdm import tqdm

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
JUDGE_INFER_DIR = os.path.dirname(SCRIPT_DIR)
RUN_IMP_DIR     = os.path.dirname(JUDGE_INFER_DIR)
sys.path.insert(0, JUDGE_INFER_DIR)

from utils_effectiveness.inference_claude import InferenceClaude
from utils_effectiveness.process_dataset import get_img_files

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "claude-sonnet-4-6"
MODEL_TAG  = MODEL_NAME.replace(".", "_").replace("-", "_")
FRAME_ROOT = "PATH/TO/VisAssist/frames"
DATA_ROOT  = os.path.join(RUN_IMP_DIR, "data_position_bias")

JUDGE_PROMPT = """\
## Role
You are a Judge and your goal is to determine which of the two responses (A or B) is more preferred (e.g., more accurate, helpful, actionable, and safety-aware) for a blind or visually impaired (BVI) user based on the provided visual information and the user's question.

## Task
I will provide you with the visual frames, the question, and two responses (Response A and Response B). Your task is to determine the preferred one, or output "Tie" if they are equally good.

## Input
Question: {question}

Response A:
{response_a}

Response B:
{response_b}

## Output Format
You must respond with ONLY one of the following three options, nothing else: "A", "B", or "Tie".

Now output your preference:\
"""

def run(trial: str):
    """
    trial: "A" (gt first) or "B" (gt second)
    """
    infer = InferenceClaude(model_name=MODEL_NAME)

    jsonl_file = os.path.join(DATA_ROOT, "position_bias_single_visassist.jsonl")
    out_dir = os.path.join(RUN_IMP_DIR, "outputs_position_single", "visassist", f"{MODEL_TAG}trial{trial}")
    os.makedirs(out_dir, exist_ok=True)

    with open(jsonl_file, encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]

    for line in tqdm(lines, desc=f"visassist/single/trial{trial}"):
        sample = json.loads(line)

        sample_id = sample["sample_id"]

        label = sample.get("label") or sample.get("failure")
        if not label:
            print(f"[Skip] {sample_id}: missing label/failure key")
            continue
        out_file = os.path.join(out_dir, f"{sample_id}_{label}.json")

        if os.path.exists(out_file):
            continue

        frames_dir = os.path.join(FRAME_ROOT, sample["frame_path"])
        img_paths = get_img_files(frames_dir)
        if not img_paths:
            print(f"[Skip] no frames: {frames_dir}")
            continue

        if "modified_answer" not in sample or "gt_answer" not in sample:
            print(f"[Skip] {sample_id}: missing gt_answer or modified_answer")
            continue

        # Trial A: gt first, Trial B: gt second
        if trial == "A":
            response_a = sample["gt_answer"]
            response_b = sample["modified_answer"]
            expected = "A"
        else:  # trial == "B"
            response_a = sample["modified_answer"]
            response_b = sample["gt_answer"]
            expected = "B"

        prompt = JUDGE_PROMPT.format(
            question=sample["question"],
            response_a=response_a,
            response_b=response_b,
        )

        try:
            raw_output = infer.processing_item(img_paths, prompt, max_new_tokens=32, temperature=0.0)
        except Exception as e:
            print(f"[Error] {sample_id}: {e}")
            continue

        # Parse the judgment result
        predicted = raw_output.strip()
        if predicted not in ["A", "B", "Tie"]:
            for choice in ["A", "B", "Tie"]:
                if choice in predicted:
                    predicted = choice
                    break

        record = {
            "sample_id":       sample_id,
            "task":            "visassist",
            "injection_type":  "single",
            "trial":           trial,
            "frame_path":      sample.get("frame_path", ""),
            "question":        sample["question"],
            "gt_answer":       sample["gt_answer"],
            "modified_answer": sample["modified_answer"],
            "gt_labels":       [label],
            "response_a":      response_a,
            "response_b":      response_b,
            "expected":        expected,
            "predicted":       predicted,
            "judge_raw_output": raw_output,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    for trial in ["A", "B"]:
        run(trial)
