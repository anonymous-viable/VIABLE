"""
Build Adversarial Injection Dataset
====================================
Reads the three dual_injected_*.jsonl files, samples 1000 unique sample_ids
per task, then for each sampled sample_id picks 3 random failure-pair
combinations.  The adversarial injection is applied to the *modified_answer*
(the one that already contains failures), NOT the gt_answer.

Output: one JSONL per task under  data_adv_injection/
  - adv_<task>.jsonl   (e.g.  adv_via_egodex.jsonl)

Each output line keeps all original fields plus:
    adv_answer, adv_injection_type
"""

import os
import sys
import json
import random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)
from adv_injector import AdversarialInjector

# ── config ────────────────────────────────────────────────────────────────────
SEED              = 42
SAMPLE_IDS_PER_TASK = 1000
COMBOS_PER_ID     = 3          # pick 3 failure combinations per sample_id
DATA_DIR          = os.path.join(ROOT_DIR, "data_effectiveness_controlled")
OUT_DIR           = SCRIPT_DIR
SOURCE_FILES = {
    "via_egodex": "dual_injected_via_egodex.jsonl",
    "visassist":  "dual_injected_visassist.jsonl",
    "walkvlm":   "dual_injected_walkvlm.jsonl",
}


def load_and_group(filepath: str) -> dict:
    """
    Load a JSONL file and group rows by sample_id.
    Returns {sample_id: [row_dict, row_dict, ...]}
    """
    groups = defaultdict(list)
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            groups[obj["sample_id"]].append(obj)
    return dict(groups)


def main():
    rng      = random.Random(SEED)
    injector = AdversarialInjector()

    os.makedirs(OUT_DIR, exist_ok=True)

    grand_total = 0
    stats = {}

    for task, fname in SOURCE_FILES.items():
        src_path = os.path.join(DATA_DIR, fname)
        print(f"\n{'='*60}")
        print(f"  Task: {task}")
        print(f"  Source: {src_path}")

        groups = load_and_group(src_path)
        all_ids = sorted(groups.keys(), key=str)
        print(f"  Total unique sample_ids: {len(all_ids)}")

        # ── sample N unique sample_ids ────────────────────────────────────
        n_sample = min(SAMPLE_IDS_PER_TASK, len(all_ids))
        sampled_ids = rng.sample(all_ids, n_sample)
        print(f"  Sampled sample_ids: {n_sample}")

        # ── for each sampled id, pick K failure combos ────────────────────
        method_counts = {m: 0 for m in AdversarialInjector.INJECTION_TYPES}
        out_path = os.path.join(OUT_DIR, f"adv_{task}.jsonl")
        row_count = 0

        with open(out_path, "w", encoding="utf-8") as fout:
            for sid in sampled_ids:
                candidates = groups[sid]
                k = min(COMBOS_PER_ID, len(candidates))
                chosen = rng.sample(candidates, k)

                for row in chosen:
                    # skip infeasible rows that lack modified_answer
                    if "modified_answer" not in row:
                        continue

                    # inject adversarial text into the MODIFIED answer
                    adv_answer, inj_type = injector.inject(
                        row["modified_answer"], method=None, rng=rng,
                    )

                    record = {
                        "sample_id":          row["sample_id"],
                        "task":               row.get("task", task),
                        "frame_path":         row.get("frame_path", ""),
                        "question":           row.get("question", ""),
                        "gt_answer":          row.get("gt_answer", ""),
                        "failure_pair":       row.get("failure_pair") or row.get("labels", []),
                        "modified_answer":    row["modified_answer"],
                        "adv_answer":         adv_answer,
                        "adv_injection_type": inj_type,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    method_counts[inj_type] += 1
                    row_count += 1

        stats[task] = {"sampled_ids": n_sample, "total_rows": row_count, "methods": method_counts}
        grand_total += row_count
        print(f"  Written: {out_path}  ({row_count} rows = {n_sample} IDs × ≤{COMBOS_PER_ID} combos)")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY   (seed={SEED})")
    print(f"{'='*60}")
    for task, s in stats.items():
        print(f"\n  [{task}]  sampled_ids={s['sampled_ids']}  total_rows={s['total_rows']}")
        for m, c in s["methods"].items():
            print(f"    {m:30s}  {c:>5d}  ({100*c/s['total_rows']:.1f}%)")
    print(f"\n  Grand total rows: {grand_total}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
