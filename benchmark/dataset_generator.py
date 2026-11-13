"""Deterministic synthetic evaluation dataset for cache hit/miss benchmarking."""

import argparse
import json
import math
import random
import re
from pathlib import Path

from benchmark.corpus import DISTINCT_PROMPTS, TOPICS

DEFAULT_OUTPUT = "benchmark/data/eval_prompts.json"


def _whitespace_case_variant(prompt: str, rng: random.Random) -> str:
    """Return the prompt (or a trivially altered copy) that still hits L1."""
    choice = rng.randrange(5)
    if choice == 0:
        return prompt
    if choice == 1:
        return prompt.lower()
    if choice == 2:
        return prompt.upper()
    if choice == 3:
        return f"  {prompt}  "
    return re.sub(r" ", "  ", prompt, count=1)


def _cycle(pool: list, n: int, rng: random.Random) -> list:
    """Pick n items, using every item before repeating (shuffled per pass)."""
    out: list = []
    while len(out) < n:
        batch = pool[:]
        rng.shuffle(batch)
        out.extend(batch)
    return out[:n]


def generate_dataset(
    total_queries: int = 200,
    l1_ratio: float = 0.30,
    l2_ratio: float = 0.40,
    miss_ratio: float = 0.30,
    seed: int = 42,
) -> dict:
    if not math.isclose(l1_ratio + l2_ratio + miss_ratio, 1.0, abs_tol=1e-9):
        raise ValueError("l1_ratio + l2_ratio + miss_ratio must sum to 1.0")
    if min(l1_ratio, l2_ratio, miss_ratio) < 0 or total_queries < 0:
        raise ValueError("ratios and total_queries must be non-negative")

    rng = random.Random(seed)

    seed_prompts = [
        {
            "id": f"seed-{i:03d}",
            "prompt": t["seed_prompt"],
            "reference_answer": t["reference_answer"],
        }
        for i, t in enumerate(TOPICS)
    ]
    topic_by_id = {s["id"]: t for s, t in zip(seed_prompts, TOPICS)}
    seed_ids = [s["id"] for s in seed_prompts]
    seed_by_id = {s["id"]: s for s in seed_prompts}

    n_l1 = round(total_queries * l1_ratio)
    n_l2 = round(total_queries * l2_ratio)
    n_miss = total_queries - n_l1 - n_l2
    n_adv = n_miss // 2
    n_distinct = n_miss - n_adv

    items: list[dict] = []

    for sid in _cycle(seed_ids, n_l1, rng):
        items.append({
            "prompt": _whitespace_case_variant(seed_by_id[sid]["prompt"], rng),
            "category": "identical",
            "expected_cache_status": "HIT-L1",
            "source_seed_id": sid,
        })

    for sid in _cycle(seed_ids, n_l2, rng):
        items.append({
            "prompt": rng.choice(topic_by_id[sid]["paraphrases"]),
            "category": "paraphrased",
            "expected_cache_status": "HIT-L2",
            "source_seed_id": sid,
        })

    for sid in _cycle(seed_ids, n_adv, rng):
        items.append({
            "prompt": rng.choice(topic_by_id[sid]["adversarial_variants"]),
            "category": "adversarial",
            "expected_cache_status": "MISS",
            "source_seed_id": sid,
        })

    for prompt in _cycle(DISTINCT_PROMPTS, n_distinct, rng):
        items.append({
            "prompt": prompt,
            "category": "distinct",
            "expected_cache_status": "MISS",
            "source_seed_id": None,
        })

    rng.shuffle(items)
    eval_queries = [{"id": f"q-{i:04d}", **item} for i, item in enumerate(items)]

    return {
        "metadata": {
            "seed": seed,
            "total_queries": total_queries,
            "ratios": {"l1": l1_ratio, "l2": l2_ratio, "miss": miss_ratio},
        },
        "seed_prompts": seed_prompts,
        "eval_queries": eval_queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--total-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = generate_dataset(total_queries=args.total_queries, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(dataset['eval_queries'])} queries, "
          f"{len(dataset['seed_prompts'])} seeds to {out}")


if __name__ == "__main__":
    main()
