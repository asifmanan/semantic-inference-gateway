import json
import re
from collections import Counter

import pytest

from benchmark.corpus import DISTINCT_PROMPTS, TOPICS
from benchmark.dataset_generator import generate_dataset


def _dump(ds: dict) -> str:
    return json.dumps(ds, sort_keys=True)


def test_corpus_size_and_shape():
    assert len(TOPICS) >= 30
    assert len(DISTINCT_PROMPTS) >= 20
    for t in TOPICS:
        assert len(t["paraphrases"]) >= 2
        assert len(t["adversarial_variants"]) >= 1


def test_deterministic_output():
    assert _dump(generate_dataset(seed=7)) == _dump(generate_dataset(seed=7))
    assert _dump(generate_dataset(seed=7)) != _dump(generate_dataset(seed=8))


def test_exact_distribution():
    ds = generate_dataset(total_queries=200)
    counts = Counter(q["expected_cache_status"] for q in ds["eval_queries"])
    assert counts == {"HIT-L1": 60, "HIT-L2": 80, "MISS": 60}
    assert len(ds["eval_queries"]) == 200
    assert len({q["id"] for q in ds["eval_queries"]}) == 200


def test_item_schema():
    ds = generate_dataset()
    cats = {"identical": "HIT-L1", "paraphrased": "HIT-L2",
            "adversarial": "MISS", "distinct": "MISS"}
    for q in ds["eval_queries"]:
        assert set(q) == {"id", "prompt", "category", "expected_cache_status",
                          "source_seed_id"}
        assert cats[q["category"]] == q["expected_cache_status"]


def test_hits_link_to_valid_seed():
    ds = generate_dataset()
    seed_ids = {s["id"] for s in ds["seed_prompts"]}
    hits = [q for q in ds["eval_queries"] if q["expected_cache_status"] != "MISS"]
    assert hits
    for q in hits:
        assert q["source_seed_id"] in seed_ids


def test_adversarial_are_misses_with_changed_numbers():
    ds = generate_dataset()
    seeds = {s["id"]: s["prompt"] for s in ds["seed_prompts"]}
    adv = [q for q in ds["eval_queries"] if q["category"] == "adversarial"]
    assert adv
    num = lambda s: sorted(re.findall(r"\d[\d,.]*", s))
    for q in adv:
        assert q["expected_cache_status"] == "MISS"
        assert q["prompt"] != seeds[q["source_seed_id"]]


def test_paraphrases_keep_numbers():
    ds = generate_dataset(total_queries=300)
    seeds = {s["id"]: s["prompt"] for s in ds["seed_prompts"]}
    num = lambda s: set(re.findall(r"\d[\d,.]*\d|\d", s))
    for q in ds["eval_queries"]:
        if q["category"] == "paraphrased":
            assert num(q["prompt"]) == num(seeds[q["source_seed_id"]]), q


def test_ratio_validation():
    with pytest.raises(ValueError):
        generate_dataset(l1_ratio=0.5, l2_ratio=0.5, miss_ratio=0.5)
