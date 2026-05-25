# VIABLE: A Visually Impaired Assistance Benchmark for VLM-as-a-Judge Evaluation

👋 Welcome to the official repository of **VIABLE**!

VIABLE is the first benchmark for evaluating VLM judge reliability in Visually Impaired Assistance (VIA), spanning three assistive scenarios, 312K+ judgment samples, and an Effectiveness–Impartiality–Stability (E-I-S) evaluation framework over a 12-mode failure taxonomy.

## 📦 Data

### `effectiveness_controlled.tar.gz`

Controlled failure-injected samples for evaluating judge effectiveness.

```bash
tar -xzf effectiveness_controlled.tar.gz
```

```
effectiveness_controlled/
├── single_injected/          # One failure per sample
│   ├── visassist.jsonl
│   ├── walkvlm.jsonl
│   └── via_egodex.jsonl
└── dual_injected/            # Two failures per sample
    ├── visassist.jsonl
    ├── walkvlm.jsonl
    └── via_egodex.jsonl
```

Each JSONL record:

| Field | Description |
|-------|-------------|
| `sample_id` | Unique identifier |
| `frame_path` | Relative path to video frames |
| `question` | User question |
| `gt_answer` | Ground-truth response |
| `modified_answer` | Response with injected failure(s) |
| `failure` / `failure_pair` | Ground-truth failure code(s) |

### 🎯 Tasks

| Task | Domain | QA Pairs |
|------|--------|----------|
| **VisAssist** | Indoor information access | 5,376 |
| **WAD (WalkVLM)** | Outdoor navigation | 2,014 |
| **VIA-EgoDex** | Egocentric manipulation | 3,205 |

### 🏷️ Failure Taxonomy

12 failure types across 4 cognitive dimensions:

| Dimension | Codes | Failure Types |
|-----------|-------|---------------|
| **Perception** | P1–P4 | Entity/Attribute Error, Spatial Mapping Error, OCR/Detail Miss, Evidence Omission |
| **Cognition** | C1–C3 | Temporal/Step Error, Causal/Functional Error, Self-Contradiction |
| **Action** | A1–A3 | Dangerous/Unsafe Advice, Vague/Non-Actionable, Incomplete Action Guidance |
| **Interaction** | I1–I2 | Redundant/Over-Verbose, Truncated/Incomplete Response |

### 📊 Benchmark Scale

| Axis | WAD | VisAssist | VIA-EgoDex | Total |
|------|-----|-----------|------------|-------|
| Effectiveness | 38,253 | 102,383 | 81,437 | 222,073 |
| Impartiality | 17,685 | 29,940 | 26,167 | 73,792 |
| Stability | 5,500 | 5,500 | 5,500 | 16,500 |
| **Total** | | | | **312,365** |

## 🚀 Usage

### 📐 Evaluate with VIABLE benchmark

```bash
git clone https://github.com/anonymous-viable/viable.git
cd viable/

# Effectiveness — failure diagnosis
python run_effectiveness/judge_infer_direct/visassist/judge_infer_visassist_openai.py
python run_effectiveness/judge_infer_direct/walkvlm/judge_infer_walkvlm_qwen_instruct.py
python run_effectiveness/judge_infer_direct/via_egodex/judge_infer_viaegodex_claude.py

# Impartiality — position bias
python run_impartiality/judge_infer_position_single/visassist/judge_infer_position_visassist_openai.py

# Impartiality — length bias
python run_impartiality/judge_infer_length/walkvlm/judge_infer_length_walkvlm_kimi_instruct.py

# Impartiality — self-preference bias
python run_impartiality/judge_infer_self_preference/via_egodex/judge_infer_self_preference_viaegodex_qwen_instruct.py

# Stability — adversarial robustness
python run_stability/judge_infer_adversarial/visassist/judge_adv_visassist_openai.py

# Stability — consistency
python run_stability/judge_infer_direct_consistency/walkvlm/judge_infer_walkvlm_internvl.py
```

### 🤖 Run VIA-Judge-Agent

```bash
git clone https://github.com/anonymous-viable/via-judge-agent.git
cd via-judge-agent/

# Run on VisAssist single-injected data
python -m agent.judge --task visassist --split single

# Run on WalkVLM dual-injected data
python -m agent.judge --task walkvlm --split dual

# Run on VIA-EgoDex (limit to 100 samples)
python -m agent.judge --task via_egodex --split single --max_samples 100
```

Results are saved to `outputs/judge_results/{task}/{split}/`.

## ⚖️ Evaluated Judge Models

| Model | Type | Scale |
|-------|------|-------|
| GPT-5.4 | Proprietary | — |
| Claude-Sonnet-4.6 | Proprietary | — |
| Qwen3-VL-8B-Instruct | Open-source | 8B |
| InternVL3.5-8B | Open-source | 8B |
| MiniCPM-V4.5-8B | Open-source | 8B |
| Kimi-VL-A3B | Open-source | 16B MoE (~3B active) |
| Youtu-VL-4B | Open-source | 4B |
