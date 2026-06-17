from __future__ import annotations

from .parser import build_parser
from .commands.config_cmds import (
    _handle_run_zero_shot_grid,
    _handle_validate_config,
)
from .commands.heatmap import (
    _handle_compare_coordinate_heatmap_scores,
    _handle_render_coordinate_heatmap,
    _handle_score_coordinate_heatmap,
    _handle_score_coordinate_heatmap_prompt_set,
)
from .commands.info import (
    _handle_demo,
    _handle_demos,
    _handle_models,
    _handle_version,
)
from .commands.manifest import (
    _handle_build_imagefolder_manifest,
    _handle_convert_manifest,
    _handle_sample_manifest,
)
from .commands.model_eval import _handle_compare_models, _handle_linear_probe
from .commands.reports import _handle_compare_reports, _handle_summarize_report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "version":
        return _handle_version()

    if args.command == "models":
        return _handle_models()

    if args.command == "demos":
        return _handle_demos()

    if args.command == "demo":
        return _handle_demo(args)

    if args.command == "convert-manifest":
        return _handle_convert_manifest(args)

    if args.command == "build-imagefolder-manifest":
        return _handle_build_imagefolder_manifest(args)

    if args.command == "sample-manifest":
        return _handle_sample_manifest(args)

    if args.command == "summarize-report":
        return _handle_summarize_report(args)

    if args.command == "compare-reports":
        return _handle_compare_reports(args)

    if args.command == "validate-config":
        return _handle_validate_config(args)

    if args.command == "run-zero-shot-grid":
        return _handle_run_zero_shot_grid(args)

    if args.command == "render-coordinate-heatmap":
        return _handle_render_coordinate_heatmap(args)

    if args.command == "score-coordinate-heatmap":
        return _handle_score_coordinate_heatmap(args)

    if args.command == "score-coordinate-heatmap-prompt-set":
        return _handle_score_coordinate_heatmap_prompt_set(args)

    if args.command == "compare-coordinate-heatmap-scores":
        return _handle_compare_coordinate_heatmap_scores(args)

    if args.command == "compare-models":
        return _handle_compare_models(args)

    if args.command == "linear-probe":
        return _handle_linear_probe(args)

    parser.print_help()
    return 0
