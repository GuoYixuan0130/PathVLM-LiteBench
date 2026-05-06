import json
from pathlib import Path

import pytest

from pathvlm_litebench.visualization.report_summary import (
    build_experiment_comparison_summary,
    build_prompt_sensitivity_experiment_summary,
    build_prompt_sensitivity_comparison_summary,
    build_retrieval_experiment_summary,
    build_retrieval_comparison_summary,
    build_zero_shot_experiment_summary,
    build_zero_shot_comparison_summary,
    save_experiment_comparison_summary,
    save_prompt_sensitivity_experiment_summary,
    save_retrieval_experiment_summary,
    save_zero_shot_experiment_summary,
)


def _write_zero_shot_report(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "metadata": {
            "model": "clip",
            "device": "cpu",
            "split": "test",
            "manifest": "dataset/MHIST/manifest.csv",
            "image_dir": "dataset/MHIST/images",
            "class_names": ["HP", "SSA"],
            "top_k": 2,
            "num_images": 4,
        },
        "metrics": {
            "classification_report": {
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "macro_precision": 0.5,
                "macro_recall": 0.5,
                "macro_f1": 0.5,
                "per_class": {
                    "HP": {
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                        "support": 2,
                    },
                    "SSA": {
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                        "support": 2,
                    },
                },
                "confusion_matrix": {
                    "class_names": ["HP", "SSA"],
                    "matrix": [[1, 1], [1, 1]],
                },
            },
            "error_summary": {
                "num_samples": 4,
                "labeled_count": 4,
                "unlabeled_count": 0,
                "num_errors": 2,
                "error_rate": 0.5,
                "true_label_distribution": {"HP": 2, "SSA": 2},
                "predicted_label_distribution": {"HP": 2, "SSA": 2},
                "warning": "Example warning.",
            },
        },
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics_payload),
        encoding="utf-8",
    )
    (report_dir / "predictions.csv").write_text(
        "image_index,image_path,true_label,predicted_label,predicted_index,confidence,correct,top_predictions_json\n"
        "0,a.png,HP,HP,0,0.8,True,[]\n"
        "1,b.png,HP,SSA,1,0.7,False,[]\n"
        "2,c.png,SSA,HP,0,0.6,False,[]\n"
        "3,d.png,SSA,SSA,1,0.9,True,[]\n",
        encoding="utf-8",
    )
    (report_dir / "errors.csv").write_text(
        "image_index,image_path,true_label,predicted_label,predicted_index,confidence,top_predictions_json\n"
        "1,b.png,HP,SSA,1,0.7,[]\n"
        "2,c.png,SSA,HP,0,0.6,[]\n",
        encoding="utf-8",
    )


def _write_retrieval_report(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "metadata": {
            "model": "clip",
            "device": "cpu",
            "split": "test",
            "manifest": "dataset/MHIST/manifest.csv",
            "image_root": "dataset/MHIST/images",
            "image_dir": None,
            "prompts": [
                "a histopathology image of hyperplastic polyp",
                "a histopathology image of sessile serrated adenoma",
            ],
            "label_prompts": ["HP", "SSA"],
            "top_k": 2,
            "recall_k": [1, 2],
            "num_images": 4,
            "num_prompts": 2,
        },
        "metrics": {
            "recall_at_k": {"R@1": 0.5, "R@2": 1.0},
            "mean_recall": 0.75,
        },
    }
    (report_dir / "retrieval_metrics.json").write_text(
        json.dumps(metrics_payload),
        encoding="utf-8",
    )
    (report_dir / "retrieval_results.csv").write_text(
        "prompt_index,prompt,target_label,rank,image_index,image_path,score,label,is_positive\n"
        "0,a histopathology image of hyperplastic polyp,HP,1,0,a.png,0.9,HP,True\n"
        "0,a histopathology image of hyperplastic polyp,HP,2,1,b.png,0.7,SSA,False\n"
        "1,a histopathology image of sessile serrated adenoma,SSA,1,1,b.png,0.8,SSA,True\n",
        encoding="utf-8",
    )


def _write_prompt_sensitivity_report(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    results = [
        {
            "concept_name": "tumor",
            "num_prompts": 2,
            "mean_topk_overlap": 0.5,
            "mean_similarity_std": 0.03,
            "prompt_results": [
                {
                    "prompt_index": 0,
                    "prompt_text": "tumor tissue",
                    "top_indices": [3, 5],
                    "top_scores": [0.9, 0.8],
                },
                {
                    "prompt_index": 1,
                    "prompt_text": "malignant tissue",
                    "top_indices": [5, 7],
                    "top_scores": [0.85, 0.75],
                },
            ],
        },
        {
            "concept_name": "normal",
            "num_prompts": 2,
            "mean_topk_overlap": 1.0,
            "mean_similarity_std": 0.01,
            "prompt_results": [],
        },
    ]
    metrics_payload = {
        "metadata": {
            "model": "clip",
            "device": "cpu",
            "image_dir": "dataset/MHIST/images",
            "top_k": 2,
            "use_pathology_prompts": True,
            "concepts": ["tumor", "normal"],
            "num_images": 4,
            "num_concepts": 2,
        },
        "results": results,
    }
    (report_dir / "prompt_sensitivity_metrics.json").write_text(
        json.dumps(metrics_payload),
        encoding="utf-8",
    )
    (report_dir / "prompt_sensitivity_summary.csv").write_text(
        "concept_name,num_prompts,mean_topk_overlap,mean_similarity_std\n"
        "tumor,2,0.5,0.03\n"
        "normal,2,1.0,0.01\n",
        encoding="utf-8",
    )
    (report_dir / "prompt_sensitivity_details.csv").write_text(
        "concept_name,prompt_index,prompt_text,rank,image_index,score\n"
        "tumor,0,tumor tissue,1,3,0.9\n"
        "tumor,0,tumor tissue,2,5,0.8\n"
        "tumor,1,malignant tissue,1,5,0.85\n"
        "tumor,1,malignant tissue,2,7,0.75\n",
        encoding="utf-8",
    )


def test_build_zero_shot_experiment_summary(tmp_path: Path):
    report_dir = tmp_path / "zero_shot_report"
    _write_zero_shot_report(report_dir)

    markdown = build_zero_shot_experiment_summary(report_dir)

    assert "# Zero-Shot Experiment Summary" in markdown
    assert "| Model | clip |" in markdown
    assert "| Accuracy | 0.5000 |" in markdown
    assert "| HP | 0.5000 | 0.5000 | 0.5000 | 2 |" in markdown
    assert "| True \\ Pred | HP | SSA |" in markdown
    assert "| predictions.csv | found (4 rows) |" in markdown
    assert "Example warning." in markdown
    assert "not clinical interpretation" in markdown


def test_save_zero_shot_experiment_summary_default_path(tmp_path: Path):
    report_dir = tmp_path / "zero_shot_report"
    _write_zero_shot_report(report_dir)

    saved_path = save_zero_shot_experiment_summary(report_dir)

    assert saved_path == str(report_dir / "experiment_summary.md")
    assert Path(saved_path).exists()
    assert "Balanced accuracy" in Path(saved_path).read_text(encoding="utf-8")


def test_save_zero_shot_experiment_summary_custom_path(tmp_path: Path):
    report_dir = tmp_path / "zero_shot_report"
    _write_zero_shot_report(report_dir)
    output_path = tmp_path / "summaries" / "summary.md"

    saved_path = save_zero_shot_experiment_summary(
        report_dir=report_dir,
        output_path=output_path,
    )

    assert saved_path == str(output_path)
    assert output_path.exists()


def test_build_zero_shot_experiment_summary_requires_metrics(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_zero_shot_experiment_summary(tmp_path / "missing_report")


def test_build_retrieval_experiment_summary(tmp_path: Path):
    report_dir = tmp_path / "retrieval_report"
    _write_retrieval_report(report_dir)

    markdown = build_retrieval_experiment_summary(report_dir)

    assert "# Retrieval Experiment Summary" in markdown
    assert "| Model | clip |" in markdown
    assert "| R@1 | 0.5000 |" in markdown
    assert "| Mean recall | 0.7500 |" in markdown
    assert "| Result rows | 3 |" in markdown
    assert "| Positive rows | 2 |" in markdown
    assert "| retrieval_results.csv | found (3 rows) |" in markdown
    assert "not clinical interpretation" in markdown


def test_save_retrieval_experiment_summary_default_path(tmp_path: Path):
    report_dir = tmp_path / "retrieval_report"
    _write_retrieval_report(report_dir)

    saved_path = save_retrieval_experiment_summary(report_dir)

    assert saved_path == str(report_dir / "experiment_summary.md")
    assert Path(saved_path).exists()
    assert "Retrieval Metrics" in Path(saved_path).read_text(encoding="utf-8")


def test_build_retrieval_experiment_summary_without_recall_metrics(tmp_path: Path):
    report_dir = tmp_path / "retrieval_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "retrieval_metrics.json").write_text(
        json.dumps(
            {
                "metadata": {"model": "clip"},
                "metrics": {"note": "Recall@K was not computed."},
            }
        ),
        encoding="utf-8",
    )

    markdown = build_retrieval_experiment_summary(report_dir)

    assert "Recall@K was not computed." in markdown
    assert "| retrieval_results.csv | missing |" in markdown


def test_build_retrieval_experiment_summary_requires_metrics(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_retrieval_experiment_summary(tmp_path / "missing_report")


def test_build_prompt_sensitivity_experiment_summary(tmp_path: Path):
    report_dir = tmp_path / "prompt_sensitivity_report"
    _write_prompt_sensitivity_report(report_dir)

    markdown = build_prompt_sensitivity_experiment_summary(report_dir)

    assert "# Prompt Sensitivity Experiment Summary" in markdown
    assert "| Model | clip |" in markdown
    assert "| tumor | 2 | 0.5000 | 0.0300 |" in markdown
    assert "| Detail rows | 4 |" in markdown
    assert "| Prompt variants in details | 2 |" in markdown
    assert "| prompt_sensitivity_details.csv | found (4 rows) |" in markdown
    assert "not clinical interpretation" in markdown
    assert "Higher mean top-k overlap" in markdown


def test_save_prompt_sensitivity_experiment_summary_default_path(tmp_path: Path):
    report_dir = tmp_path / "prompt_sensitivity_report"
    _write_prompt_sensitivity_report(report_dir)

    saved_path = save_prompt_sensitivity_experiment_summary(report_dir)

    assert saved_path == str(report_dir / "experiment_summary.md")
    assert Path(saved_path).exists()
    assert "Concept Summary" in Path(saved_path).read_text(encoding="utf-8")


def test_build_prompt_sensitivity_experiment_summary_without_csv(tmp_path: Path):
    report_dir = tmp_path / "prompt_sensitivity_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "prompt_sensitivity_metrics.json").write_text(
        json.dumps(
            {
                "metadata": {"model": "clip"},
                "results": [
                    {
                        "concept_name": "tumor",
                        "num_prompts": 2,
                        "mean_topk_overlap": 0.5,
                        "mean_similarity_std": 0.03,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    markdown = build_prompt_sensitivity_experiment_summary(report_dir)

    assert "| tumor | 2 | 0.5000 | 0.0300 |" in markdown
    assert "| prompt_sensitivity_summary.csv | missing |" in markdown


def test_build_prompt_sensitivity_experiment_summary_requires_metrics(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_prompt_sensitivity_experiment_summary(tmp_path / "missing_report")


def test_build_zero_shot_comparison_summary(tmp_path: Path):
    clip_report = tmp_path / "clip_zero_shot"
    plip_report = tmp_path / "plip_zero_shot"
    _write_zero_shot_report(clip_report)
    _write_zero_shot_report(plip_report)

    payload = json.loads((plip_report / "metrics.json").read_text(encoding="utf-8"))
    payload["metadata"]["model"] = "plip"
    payload["metrics"]["classification_report"]["accuracy"] = 0.75
    payload["metrics"]["classification_report"]["balanced_accuracy"] = 0.8
    payload["metrics"]["classification_report"]["macro_f1"] = 0.7
    payload["metrics"]["error_summary"]["predicted_label_distribution"] = {
        "SSA": 3,
        "HP": 1,
    }
    (plip_report / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    markdown = build_zero_shot_comparison_summary(
        [clip_report, plip_report],
        run_names=["CLIP default", "PLIP default"],
    )

    assert "# Zero-Shot Comparison Summary" in markdown
    assert "| CLIP default | clip | test | 4 | 0.5000 | 0.5000 | 0.5000 |" in markdown
    assert "| PLIP default | plip | test | 4 | 0.7500 | 0.8000 | 0.7000 |" in markdown
    assert "HP=1, SSA=3" in markdown
    assert "| CLIP default |" in markdown
    assert "not clinical interpretation" in markdown


def test_build_retrieval_comparison_summary(tmp_path: Path):
    clip_report = tmp_path / "clip_retrieval"
    plip_report = tmp_path / "plip_retrieval"
    _write_retrieval_report(clip_report)
    _write_retrieval_report(plip_report)

    payload = json.loads((plip_report / "retrieval_metrics.json").read_text(encoding="utf-8"))
    payload["metadata"]["model"] = "plip"
    payload["metrics"]["recall_at_k"]["R@1"] = 1.0
    payload["metrics"]["mean_recall"] = 1.0
    (plip_report / "retrieval_metrics.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    markdown = build_retrieval_comparison_summary([clip_report, plip_report])

    assert "# Retrieval Comparison Summary" in markdown
    assert "| Run | Model | Split | Images | Prompts | R@1 | R@2 | Mean recall | Note |" in markdown
    assert "| clip_retrieval | clip | test | 4 | 2 | 0.5000 | 1.0000 | 0.7500 |  |" in markdown
    assert "| plip_retrieval | plip | test | 4 | 2 | 1.0000 | 1.0000 | 1.0000 |  |" in markdown


def test_build_prompt_sensitivity_comparison_summary(tmp_path: Path):
    first_report = tmp_path / "prompt_a"
    second_report = tmp_path / "prompt_b"
    _write_prompt_sensitivity_report(first_report)
    _write_prompt_sensitivity_report(second_report)

    payload = json.loads(
        (second_report / "prompt_sensitivity_metrics.json").read_text(encoding="utf-8")
    )
    payload["metadata"]["model"] = "plip"
    payload["results"][0]["mean_topk_overlap"] = 0.25
    payload["results"][1]["mean_topk_overlap"] = 0.75
    (second_report / "prompt_sensitivity_metrics.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    markdown = build_prompt_sensitivity_comparison_summary(
        [first_report, second_report],
        run_names=["prompt set A", "prompt set B"],
    )

    assert "# Prompt Sensitivity Comparison Summary" in markdown
    assert "| prompt set A | clip | 4 | 2 | 0.7500 | 0.0200 | True |" in markdown
    assert "| prompt set B | plip | 4 | 2 | 0.5000 | 0.0200 | True |" in markdown
    assert "averaged across concepts" in markdown


def test_build_experiment_comparison_summary_dispatches(tmp_path: Path):
    report_dir = tmp_path / "zero_shot_report"
    _write_zero_shot_report(report_dir)

    markdown = build_experiment_comparison_summary("zero-shot", [report_dir])

    assert "# Zero-Shot Comparison Summary" in markdown


def test_build_experiment_comparison_summary_rejects_unknown_task(tmp_path: Path):
    with pytest.raises(ValueError):
        build_experiment_comparison_summary("unknown", [tmp_path])


def test_build_comparison_summary_requires_report_dirs():
    with pytest.raises(ValueError):
        build_zero_shot_comparison_summary([])


def test_build_comparison_summary_requires_matching_run_names(tmp_path: Path):
    report_dir = tmp_path / "zero_shot_report"
    _write_zero_shot_report(report_dir)

    with pytest.raises(ValueError):
        build_zero_shot_comparison_summary([report_dir], run_names=["a", "b"])


def test_save_experiment_comparison_summary(tmp_path: Path):
    first_report = tmp_path / "clip_zero_shot"
    second_report = tmp_path / "plip_zero_shot"
    _write_zero_shot_report(first_report)
    _write_zero_shot_report(second_report)
    output_path = tmp_path / "comparison" / "comparison_summary.md"

    saved_path = save_experiment_comparison_summary(
        task="zero-shot",
        report_dirs=[first_report, second_report],
        output_path=output_path,
        run_names=["clip", "plip"],
    )

    assert saved_path == str(output_path)
    assert output_path.exists()
    assert "Zero-Shot Comparison Summary" in output_path.read_text(encoding="utf-8")
