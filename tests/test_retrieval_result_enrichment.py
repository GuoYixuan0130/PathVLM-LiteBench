import importlib.util
from pathlib import Path

import pytest


def load_retrieval_demo_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "01_patch_text_retrieval_demo.py"
    spec = importlib.util.spec_from_file_location("retrieval_demo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load retrieval demo module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_enrich_retrieval_results_with_labels_basic():
    module = load_retrieval_demo_module()

    retrieval_results = [
        [
            {"index": 0, "score": 0.9, "path": "a.png"},
            {"index": 1, "score": 0.8, "path": "b.png"},
        ],
        [
            {"index": 2, "score": 0.7, "path": "c.png"},
        ],
    ]
    labels = ["HP", "SSA", "HP"]
    label_prompts = ["HP", "SSA"]

    enriched = module.enrich_retrieval_results_with_labels(
        retrieval_results=retrieval_results,
        labels=labels,
        label_prompts=label_prompts,
    )

    assert enriched[0][0]["label"] == "HP"
    assert enriched[0][0]["target_label"] == "HP"
    assert enriched[0][0]["is_positive"] is True
    assert enriched[0][1]["label"] == "SSA"
    assert enriched[0][1]["is_positive"] is False
    assert enriched[1][0]["target_label"] == "SSA"
    assert enriched[1][0]["is_positive"] is False

    assert "label" not in retrieval_results[0][0]


def test_enrich_retrieval_results_with_labels_no_labels():
    module = load_retrieval_demo_module()

    retrieval_results = [[{"index": 0, "score": 0.9, "path": "a.png"}]]
    enriched = module.enrich_retrieval_results_with_labels(
        retrieval_results=retrieval_results,
        labels=None,
        label_prompts=["HP"],
    )

    assert enriched == retrieval_results
    assert "label" not in enriched[0][0]


def test_enrich_retrieval_results_with_labels_prompt_length_mismatch():
    module = load_retrieval_demo_module()

    retrieval_results = [[{"index": 0, "score": 0.9, "path": "a.png"}]]
    labels = ["HP"]

    with pytest.raises(ValueError):
        module.enrich_retrieval_results_with_labels(
            retrieval_results=retrieval_results,
            labels=labels,
            label_prompts=["HP", "SSA"],
        )


def test_enrich_retrieval_results_with_labels_index_out_of_range():
    module = load_retrieval_demo_module()

    retrieval_results = [[{"index": 2, "score": 0.9, "path": "a.png"}]]
    labels = ["HP", "SSA"]

    with pytest.raises(ValueError):
        module.enrich_retrieval_results_with_labels(
            retrieval_results=retrieval_results,
            labels=labels,
            label_prompts=["HP"],
        )


def test_enrich_retrieval_results_with_labels_none_label_sets_negative_match():
    module = load_retrieval_demo_module()

    retrieval_results = [[{"index": 0, "score": 0.9, "path": "a.png"}]]
    labels = [None]

    enriched = module.enrich_retrieval_results_with_labels(
        retrieval_results=retrieval_results,
        labels=labels,
        label_prompts=["HP"],
    )

    assert enriched[0][0]["label"] is None
    assert enriched[0][0]["target_label"] == "HP"
    assert enriched[0][0]["is_positive"] is False
