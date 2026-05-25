"""
Downstream Validation: Judge Feedback → Generator Revision

Step 3: Generator revises its initial response using judge feedback.
        Produces revised responses for both raw and harnessed feedback conditions.
"""

import json
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "PATH/TO/Qwen3-VL-8B-Instruct"
INPUT_FILE = "outputs/step2_judge_feedback/judge_feedback.jsonl"
OUTPUT_DIR = "outputs/step3_revised_responses"
FRAME_ROOTS = {
    "visassist":  "PATH/TO/VisAssist/frames",
    "walkvlm":    "PATH/TO/WalkVLM/frames",
    "via_egodex": "PATH/TO/EgoDex/frames",
}
MAX_NEW_TOKENS = 256

REVISION_PROMPT_TEMPLATE = """\
You are revising an answer for a visually impaired user.
The original answer has been reviewed and feedback is provided below.
Please produce an improved answer that addresses the feedback.

[Question]:
{question}

[Original Answer]:
{initial_response}

[Feedback]:
{feedback}

Please revise the answer to be more accurate, complete, and useful.
Keep it concise (under 256 tokens). Output ONLY the revised answer.
"""


# ── Model loading ─────────────────────────────────────────────────────────────
def load_generator(model_path):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"[Generator] Loading {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()
    print("[Generator] Loaded.")
    return model, processor


@torch.inference_mode()
def revise_response(model, processor, img_paths, question, initial_response, feedback, max_new_tokens=256):
    """Generate a revised response given images, question, initial answer, and feedback."""
    revision_text = REVISION_PROMPT_TEMPLATE.format(
        question=question,
        initial_response=initial_response,
        feedback=feedback,
    )

    content = []
    for p in img_paths[:4]:
        content.append({"type": "image", "image": p})
    content.append({"type": "text", "text": revision_text})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    tok = processor.tokenizer
    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def get_frame_paths(item):
    task = item["task"]
    frame_root = FRAME_ROOTS[task]
    frames_dir = os.path.join(frame_root, item["frame_path"])
    if not os.path.isdir(frames_dir):
        return []
    files = [f for f in os.listdir(frames_dir)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    def numeric_key(name):
        stem = os.path.splitext(name)[0]
        return int(stem) if stem.isdigit() else stem
    files.sort(key=numeric_key)
    return [os.path.join(frames_dir, f) for f in files]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.path.dirname(__file__), input_path)

    with open(input_path) as f:
        items = [json.loads(line) for line in f if line.strip()]

    if args.max_samples:
        items = items[:args.max_samples]

    print(f"Loaded {len(items)} samples")

    # Load generator
    model, processor = load_generator(args.model)

    output_file = os.path.join(args.output_dir, "revised_responses.jsonl")
    with open(output_file, "w") as fout:
        for idx, item in enumerate(items):
            frame_paths = get_frame_paths(item)
            question = item["question"]
            initial_response = item["initial_response"]

            print(f"\n[{idx}/{len(items)}] {item['sample_id']} ...")

            # Condition 1: No feedback (just keep initial)
            no_feedback_response = initial_response

            # Condition 2: Raw judge feedback → revision
            raw_feedback = item["raw_judge"]["feedback"]
            print("  Revising with raw feedback...")
            raw_revised = revise_response(
                model, processor, frame_paths,
                question, initial_response, raw_feedback, MAX_NEW_TOKENS
            )

            # Condition 3: Harnessed judge feedback → revision
            harnessed_feedback = item["harnessed_judge"]["feedback"]
            print("  Revising with harnessed feedback...")
            harnessed_revised = revise_response(
                model, processor, frame_paths,
                question, initial_response, harnessed_feedback, MAX_NEW_TOKENS
            )

            result = {
                "sample_id": item["sample_id"],
                "task": item["task"],
                "frame_path": item["frame_path"],
                "question": question,
                "gt_answer": item.get("gt_answer", ""),
                "initial_response": initial_response,
                "raw_judge_feedback": raw_feedback,
                "raw_judge_codes": item["raw_judge"]["failure_codes"],
                "harnessed_judge_feedback": harnessed_feedback,
                "harnessed_judge_codes": item["harnessed_judge"]["failure_codes"],
                "revised_no_feedback": no_feedback_response,
                "revised_raw_feedback": raw_revised,
                "revised_harnessed_feedback": harnessed_revised,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"  Raw revised: {raw_revised[:80]}...")
            print(f"  Harnessed revised: {harnessed_revised[:80]}...")

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
