"""Command-line interface for extraction and reproducible benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .accessibility import write_accessible_html
from .benchmark import run_benchmark
from .models import ExtractorConfig, NormalizedROI
from .pipeline import HeadlineExtractor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-ocr",
        description="Extract static lower-thirds and assemble scrolling news tickers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract a headline from one video")
    extract.add_argument("video", type=Path)
    extract.add_argument("--roi", default="0.02,0.64,0.96,0.32")
    extract.add_argument("--sample-fps", type=float, default=2.0)
    extract.add_argument("--mode", choices=("auto", "static", "ticker"), default="auto")
    extract.add_argument("--output", type=Path, default=Path("outputs/headline.json"))
    extract.add_argument("--accessible-html", type=Path)

    benchmark = subparsers.add_parser("benchmark", help="run the deterministic ablation benchmark")
    benchmark.add_argument("--artifacts", type=Path, default=Path("benchmark_artifacts"))
    benchmark.add_argument("--output", type=Path, default=Path("results"))
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--regenerate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "extract":
        config = ExtractorConfig(
            roi=NormalizedROI.parse(args.roi),
            sample_fps=args.sample_fps,
            mode=args.mode,
        )
        result = HeadlineExtractor(config=config).extract(args.video)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        if args.accessible_html:
            write_accessible_html(result, args.accessible_html)
        print(result.headline or "No headline detected")
        print(f"JSON: {args.output.resolve()}")
        return 0 if result.detected else 2

    report = run_benchmark(
        args.artifacts,
        args.output,
        limit=args.limit,
        regenerate=args.regenerate,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
