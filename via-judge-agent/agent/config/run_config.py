"""
Configuration for VIA-Judge-Agent.
"""

import os

# ── Model backend ─────────────────────────────────────────────────────────
# "openai" → GPT-5.4 via API;  "qwen" → local Qwen-VL
BACKEND = os.environ.get("JUDGE_BACKEND", "openai")

# OpenAI settings
OPENAI_MODEL = "gpt-5.4"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)

# Local Qwen-VL settings
QWEN_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "MODEL/PATH/TO/Qwen3-VL-8B-Instruct")

# ── Data paths ────────────────────────────────────────────────────────────
FRAME_ROOTS = {
    "visassist":  "DATA/PATH/TO/VisAssist/frames",
    "walkvlm":    "DATA/PATH/TO/WalkVLM/frames",
    "via_egodex": "DATA/PATH/TO/EgoDex/frames",
}

DATA_ROOT = "./data_one_fifth/per_situation"

# ── Model local paths ─────────────────────────────────────────────────────
MODEL_ROOT = "MODEL/PATH"
GROUNDING_DINO_PATH = os.path.join(MODEL_ROOT, "IDEA-Research/grounding-dino-base")
SAM2_CHECKPOINT = os.path.join(MODEL_ROOT, "facebook/sam2.1-hiera-large")
DEPTH_MODEL_PATH = os.path.join(MODEL_ROOT, "depth-anything/DA3-LARGE-1.1")

# ── Evidence extractor settings ───────────────────────────────────────────
DEVICE = "cuda:0"

# ── Judge hyperparameters ─────────────────────────────────────────────────
TOP_K = 2               # output at most top-k failure codes
CONFIDENCE_THRESHOLD = 0.5  # minimum confidence to count as FAIL
