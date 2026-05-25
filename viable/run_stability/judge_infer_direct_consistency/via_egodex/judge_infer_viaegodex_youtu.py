import sys
import os
import json
from tqdm import tqdm

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
JUDGE_INFER_DIR = os.path.dirname(SCRIPT_DIR)
RUN_EFF_DIR     = os.path.dirname(JUDGE_INFER_DIR)
sys.path.insert(0, JUDGE_INFER_DIR)

from utils_effectiveness.inference_youtu import InferenceYoutu
from utils_effectiveness.process_dataset import get_img_files

# ── config ────────────────────────────────────────────────────────────────────
MODEL_LOCAL_PATH = "PATH/TO/models/tencent/Youtu-VL-4B-Instruct"
MODEL_TAG  = "youtu_vl_4b_instruct"
FRAME_ROOT = "PATH/TO/EgoDex/frames"
DATA_ROOT  = os.path.join(RUN_EFF_DIR, "data_effectiveness_controlled")
NUM_RUNS = 3
NUM_SAMPLE_IDS = 500

JUDGE_PROMPT = """\
You are a judge assessing the quality of AI vision assistant responses for blind or visually impaired users. Your task is to diagnose any failures in the candidate response based on the vision images and the user's question.

According to the following taxonomy, identify the single most likely failure type.

Failure types:
P1 - Entity/Attribute Error: describes non-existent object or wrong attribute/quantity/state
P2 - Spatial Mapping Error: wrong position, direction, or spatial relationship (left/right, front/back)
P3 - OCR/Detail Miss: misreads text in frame or misses small critical visual cue
P4 - Evidence Omission: critical visual info is visible but not described or flagged
C1 - Temporal/Step Error: missing step or wrong order in action sequence
C2 - Unjustified Inference: conclusion beyond visible evidence, stated as fact without uncertainty
C3 - Internal Contradiction: two statements in same answer conflict each other
A1 - Safety Violation: advice poses physical risk or omits necessary safety warning
A2 - Non-actionable Guidance: no concrete action pointer, user doesn't know what to do
A3 - Proactive Clarification Failure: gives definitive answer when image is blurry/ambiguous
I1 - Redundant Output: substantial irrelevant content, forces user to listen longer
I2 - Truncated Output: answer cut off before complete
N  - No Significant Shortcomings: answer is accurate, safe, and well-communicated

The blind user asked:
{question}

Candidate response:
"{candidate}"

Instructions:
- Compare the candidate response against the image content and the user's question.
- Identify only the single most likely failure type.
- If no significant failure exists, output N.
- Reply with ONLY a JSON list containing exactly one failure code — no explanation, no extra text.

Examples of valid output:
["P1"]
["A1"]
["N"]\
"""

def run(run_id: int = 0):
    infer  = InferenceYoutu(MODEL_LOCAL_PATH)

    jsonl_file = os.path.join(DATA_ROOT, f"single_injected_via_egodex.jsonl")
    out_dir    = os.path.join(RUN_EFF_DIR, "outputs_stable", f"output_stable_run{run_id}_single_injected_via_egodex_{MODEL_TAG}")
    os.makedirs(out_dir, exist_ok=True)

    with open(jsonl_file, encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]

    # Filter the first NUM_SAMPLE_IDS unique sample_ids (keep all failure types for each id)
    selected_ids = set()
    filtered_lines = []
    for line in lines:
        sid = json.loads(line)["sample_id"]
        if sid not in selected_ids:
            if len(selected_ids) >= NUM_SAMPLE_IDS:
                continue
            selected_ids.add(sid)
        filtered_lines.append(line)
    print(f"Selected {len(selected_ids)} sample_ids, {len(filtered_lines)} total entries")

    for line in tqdm(filtered_lines, desc=f"via_egodex/single_injected"):
        sample = json.loads(line)

        # Get sample_id and label
        sample_id = sample["sample_id"]
        label = sample.get("label")
        if not label:
            print(f"[Skip] {sample_id}: missing 'label' key")
            continue
        out_file = os.path.join(out_dir, f"{sample_id}_{label}.json")

        if os.path.exists(out_file):
            continue

        frames_dir = os.path.join(FRAME_ROOT, sample["frame_path"])
        img_paths  = get_img_files(frames_dir)
        if not img_paths:
            print(f"[Skip] no frames: {frames_dir}")
            continue

        if "modified_answer" not in sample:
            print(f"[Skip] {sample_id}: missing 'modified_answer' key")
            continue

        prompt = JUDGE_PROMPT.format(
            question=sample["question"],
            candidate=sample["modified_answer"],
        )

        try:
            raw_output = infer.processing_item(img_paths, prompt, max_new_tokens=128)
        except Exception as e:
            print(f"[Error] {sample_id}: {e}")
            continue

        # Parse failure codes
        predicted_codes = []
        try:
            parsed = json.loads(raw_output.strip())
            if isinstance(parsed, list):
                predicted_codes = parsed
        except:
            pass

        gt_labels = [sample.get("label")]

        record = {
            "sample_id":        sample_id,
            "run_id":           run_id,
            "task":             "via_egodex",
            "injection_type":   "single_injected",
            "frame_path":       sample.get("frame_path", ""),
            "question":         sample["question"],
            "gt_answer":        sample.get("gt_answer", ""),
            "modified_answer":  sample["modified_answer"],
            "gt_labels":        gt_labels,
            "predicted_codes":  predicted_codes,
            "judge_raw_output": raw_output,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    for r in range(NUM_RUNS):
        print(f"\n{'='*60}\n  Stability run {r+1}/{NUM_RUNS}\n{'='*60}")
        run(run_id=r)
