from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _resolve_compare_models_class_names(
    args: argparse.Namespace,
    records,
) -> list[str]:
    from ...data import get_unique_labels

    if args.class_names is not None:
        return list(args.class_names)

    unique_labels = get_unique_labels(records)
    if not unique_labels:
        raise ValueError("Manifest has no labels; provide --class-names.")
    if any(label.strip().isdigit() for label in unique_labels):
        raise ValueError(
            "Manifest labels look like integer class indices; pass --class-names "
            "in class-index order so prompts can be built."
        )
    return unique_labels


def _resolve_compare_models_prompts(
    args: argparse.Namespace,
    class_names: list[str],
) -> list[str]:
    if args.class_prompts is not None:
        if len(args.class_prompts) != len(class_names):
            raise ValueError(
                f"--class-prompts count ({len(args.class_prompts)}) must match "
                f"the number of classes ({len(class_names)})."
            )
        return list(args.class_prompts)

    if "{}" not in args.prompt_template:
        raise ValueError(
            "--prompt-template must contain a '{}' slot for the class name, "
            "or pass explicit --class-prompts instead."
        )
    return [args.prompt_template.format(name) for name in class_names]


def _handle_compare_models(args: argparse.Namespace) -> int:
    from ...data import (
        filter_records_by_split,
        load_patch_manifest,
        records_to_image_paths,
        records_to_labels,
    )

    try:
        records = load_patch_manifest(
            manifest_path=args.manifest,
            image_root=args.image_root,
            require_exists=True,
        )
        if args.split is not None:
            records = filter_records_by_split(records, args.split)
            if not records:
                raise ValueError(f"No manifest records matched split '{args.split}'.")
        if args.max_images is not None:
            records = records[: args.max_images]

        class_names = _resolve_compare_models_class_names(args, records)
        class_prompts = _resolve_compare_models_prompts(args, class_names)

        from ...evaluation import resolve_true_indices

        labels = records_to_labels(records)
        true_indices = resolve_true_indices(labels, class_names)

        output_dir = Path(args.output_dir)
        csv_path = output_dir / "model_comparison.csv"
        per_class_csv_path = output_dir / "model_comparison_per_class.csv"
        chart_path = output_dir / "model_comparison.png"
        metadata_path = output_dir / "metadata.json"

        if args.dry_run:
            print("Dry run only. No model inference was run.")
            print(f"Manifest: {args.manifest}")
            print(f"Patches: {len(records)}")
            print(f"Models: {', '.join(args.models)}")
            print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
            print("Prompts:")
            for prompt in class_prompts:
                print(f"  - {prompt}")
            print(f"CSV output: {csv_path}")
            print(f"Per-class CSV output: {per_class_csv_path}")
            print(f"Chart output: {chart_path}")
            print(f"Metadata output: {metadata_path}")
            return 0

        from ...data import load_patch_images_from_paths
        from ...environment import collect_environment
        from ...evaluation import evaluate_models_zero_shot
        from ...visualization import (
            compute_model_accuracy_cis,
            save_model_comparison_chart,
            save_model_comparison_csv,
            save_model_comparison_per_class_csv,
        )

        image_paths = records_to_image_paths(records)
        images, _ = load_patch_images_from_paths(image_paths)

        results = evaluate_models_zero_shot(
            images,
            true_indices,
            class_prompts,
            args.models,
            device=args.device,
            batch_size=args.batch_size,
        )

        cis = compute_model_accuracy_cis(
            results,
            confidence=args.confidence,
            num_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

        random_baseline = 1.0 / len(class_names)
        subtitle = (
            f"{len(images)} patches · {len(class_names)} classes · frozen · "
            f"shared prompt template"
        )
        save_model_comparison_csv(results, csv_path, cis=cis)
        save_model_comparison_per_class_csv(results, class_names, per_class_csv_path)
        save_model_comparison_chart(
            results,
            chart_path,
            title=args.title,
            subtitle=subtitle,
            random_baseline=random_baseline,
            cis=cis,
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "manifest": args.manifest,
                    "num_images": len(images),
                    "models": list(args.models),
                    "class_names": class_names,
                    "class_prompts": class_prompts,
                    "prompt_template": (
                        None if args.class_prompts is not None else args.prompt_template
                    ),
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "split": args.split,
                    "random_baseline": random_baseline,
                    "bootstrap": {
                        "confidence": args.confidence,
                        "num_resamples": args.bootstrap_resamples,
                        "seed": args.seed,
                    },
                    "environment": collect_environment(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "results": [
                        {
                            "model": result.model,
                            "accuracy": result.accuracy,
                            "accuracy_ci": cis[result_index],
                            "correct": result.correct,
                            "total": result.total,
                            "per_class": [
                                {
                                    "class_index": index,
                                    "class_name": class_names[index],
                                    "correct": result.per_class_correct[index],
                                    "total": result.per_class_total[index],
                                    "accuracy": (
                                        None
                                        if result.per_class_total[index] == 0
                                        else result.per_class_correct[index]
                                        / result.per_class_total[index]
                                    ),
                                }
                                for index in range(len(class_names))
                            ],
                        }
                        for result_index, result in enumerate(results)
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved model comparison CSV to: {csv_path}")
    print(f"Saved per-class comparison CSV to: {per_class_csv_path}")
    print(f"Saved model comparison chart to: {chart_path}")
    print(f"Saved model comparison metadata to: {metadata_path}")
    print(f"Patches: {len(images)}")
    for result, ci in zip(results, cis):
        line = f"- {result.model}: {result.accuracy:.1%} ({result.correct}/{result.total})"
        if ci is not None:
            line += (
                f"  {ci['confidence']:.0%} CI [{ci['ci_low']:.1%}, {ci['ci_high']:.1%}]"
            )
        print(line)
    return 0


def _handle_linear_probe(args: argparse.Namespace) -> int:
    from ...data import (
        filter_records_by_split,
        load_patch_manifest,
        records_to_image_paths,
        records_to_labels,
    )

    try:
        records = load_patch_manifest(
            manifest_path=args.manifest,
            image_root=args.image_root,
            require_exists=True,
        )
        train_records = filter_records_by_split(records, args.train_split)
        test_records = filter_records_by_split(records, args.test_split)
        if not train_records:
            raise ValueError(
                f"No manifest records matched train split '{args.train_split}'."
            )
        if not test_records:
            raise ValueError(
                f"No manifest records matched test split '{args.test_split}'."
            )
        if args.max_images is not None:
            train_records = train_records[: args.max_images]
            test_records = test_records[: args.max_images]

        train_labels = records_to_labels(train_records)
        if any(label is None or not str(label).strip() for label in train_labels):
            raise ValueError(
                "Every train record must be labeled to fit a linear probe."
            )
        test_labels = records_to_labels(test_records)

        output_dir = Path(args.output_dir)
        predictions_path = output_dir / "predictions.csv"
        errors_path = output_dir / "errors.csv"
        metrics_path = output_dir / "metrics.json"

        if args.dry_run:
            print("Dry run only. No model inference was run.")
            print(f"Manifest: {args.manifest}")
            print(f"Model: {args.model}")
            print(f"Train split '{args.train_split}': {len(train_records)} patches")
            print(f"Test split '{args.test_split}': {len(test_records)} patches")
            print(f"Predictions output: {predictions_path}")
            print(f"Metrics output: {metrics_path}")
            return 0

        from ...data import load_patch_images_from_paths
        from ...environment import collect_environment
        from ...evaluation import (
            accuracy_ci_from_labels,
            compute_classification_report,
            run_linear_probe,
        )
        from ...models import create_model
        from ...visualization import (
            save_classification_metrics_json,
            save_zero_shot_errors_csv,
            save_zero_shot_predictions_csv,
        )

        model = create_model(args.model, args.device)
        train_images, _ = load_patch_images_from_paths(
            records_to_image_paths(train_records)
        )
        test_images, test_image_paths = load_patch_images_from_paths(
            records_to_image_paths(test_records)
        )

        train_embeddings = model.encode_images(train_images, batch_size=args.batch_size)
        test_embeddings = model.encode_images(test_images, batch_size=args.batch_size)

        probe = run_linear_probe(
            train_embeddings,
            train_labels,
            test_embeddings,
            class_names=args.class_names,
            C=args.C,
            max_iter=args.max_iter,
            seed=args.seed,
            normalize=not args.no_normalize,
        )

        predicted_labels = probe["predicted_labels"]
        results = [
            {
                "image_index": index,
                "predicted_label": predicted_labels[index],
                "predicted_index": probe["predicted_indices"][index],
                "confidence": probe["confidences"][index],
                "top_predictions": [],
            }
            for index in range(len(predicted_labels))
        ]

        labeled_pairs = [
            (str(true), predicted_labels[index])
            for index, true in enumerate(test_labels)
            if true is not None and str(true).strip()
        ]
        if not labeled_pairs:
            raise ValueError(
                "No labeled test records; cannot evaluate the linear probe."
            )
        labeled_true = [pair[0] for pair in labeled_pairs]
        labeled_pred = [pair[1] for pair in labeled_pairs]

        report = compute_classification_report(
            labeled_true,
            labeled_pred,
            class_names=args.class_names,
        )
        report["accuracy_ci"] = accuracy_ci_from_labels(
            test_labels,
            predicted_labels,
            confidence=args.confidence,
            num_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

        save_zero_shot_predictions_csv(
            test_image_paths, results, predictions_path, true_labels=test_labels
        )
        save_zero_shot_errors_csv(
            test_image_paths, results, errors_path, true_labels=test_labels
        )
        save_classification_metrics_json(
            report,
            metrics_path,
            metadata={
                "task": "linear-probe",
                "manifest": args.manifest,
                "model": args.model,
                "device": args.device,
                "train_split": args.train_split,
                "test_split": args.test_split,
                "num_train": probe["num_train"],
                "num_test": probe["num_test"],
                "embedding_dim": probe["embedding_dim"],
                "probe": {
                    "classifier": "logistic_regression",
                    "C": probe["C"],
                    "max_iter": probe["max_iter"],
                    "normalize": probe["normalize"],
                    "seed": probe["seed"],
                },
                "bootstrap": {
                    "confidence": args.confidence,
                    "num_resamples": args.bootstrap_resamples,
                    "seed": args.seed,
                },
                "environment": collect_environment(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    accuracy_ci = report["accuracy_ci"]
    print(f"Saved linear-probe predictions to: {predictions_path}")
    print(f"Saved linear-probe errors to: {errors_path}")
    print(f"Saved linear-probe metrics to: {metrics_path}")
    print(f"Train patches: {probe['num_train']} · Test patches: {probe['num_test']}")
    print(
        f"Accuracy: {report['accuracy']:.1%} "
        f"({accuracy_ci['confidence']:.0%} CI "
        f"[{accuracy_ci['ci_low']:.1%}, {accuracy_ci['ci_high']:.1%}])"
    )
    print(f"Balanced accuracy: {report['balanced_accuracy']:.1%}")
    print(f"Macro F1: {report['macro_f1']:.3f}")
    return 0
