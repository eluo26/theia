"""Command line for the vision module.

  python -m vision.cli check-config
  python -m vision.cli describe path/to/image.jpg
  python -m vision.cli index SCAN_DIR
  python -m vision.cli index video.mp4 --sweep START_PAN END_PAN TILT
  python -m vision.cli index video.mp4 --angles-csv angles.csv
  python -m vision.cli ask SCAN_DIR "where's my blue water bottle?"
  python -m vision.cli answer SCAN_DIR "where's my blue water bottle?"
  python -m vision.cli eval desk
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vision.config import MissingAPIKeyError, api_key_is_present, load_settings, require_api_key
from vision.eval import EvalError, evaluate_scene
from vision.grok_client import GrokCallError, describe_image, redact
from vision.index import build_catalog, catalog_for_query
from vision.ingest import IngestError
from vision.integrate import answer
from vision.query import locate, public_query_dict
from vision.viz import annotate_image, legend_name

logger = logging.getLogger("vision.cli")


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(
                redact(item) if isinstance(item, str) else item for item in record.args
            )
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_RedactFilter())
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vision.cli",
        description="Object localization for the laser-pointing turret.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-config", help="Print loaded settings. Does not print the API key.")

    describe = sub.add_parser(
        "describe",
        help="Run one vision call on one image and write an annotated debug image.",
    )
    describe.add_argument(
        "image",
        type=Path,
        help="A .jpg, .jpeg, .png, .webp, .heic, or non-animated .gif file. HEIC is converted to JPEG before the vision call.",
    )
    describe.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the JSON and annotated image (default: data/out)",
    )
    describe.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the on-disk response cache and call the API",
    )

    index = sub.add_parser(
        "index",
        help="Build a catalog from angle-tagged images or a sweep video.",
    )
    index.add_argument("scan_dir", type=Path, help="Image folder or a video file")
    index.add_argument(
        "--sweep",
        nargs=3,
        type=float,
        metavar=("START_PAN", "END_PAN", "TILT"),
        help="Constant-speed video sweep: start pan, end pan, and a fixed tilt",
    )
    index.add_argument(
        "--angles-csv",
        type=Path,
        help="Video sidecar CSV with columns timestamp_s,pan,tilt",
    )
    index.add_argument("--out-dir", type=Path, default=None)
    index.add_argument("--no-cache", action="store_true")
    index.add_argument(
        "--debug",
        action="store_true",
        help="Write annotated images under the output debug folder",
    )

    ask = sub.add_parser("ask", help="Index a scan and print a QueryResult.")
    ask.add_argument("scan_dir", type=Path, help="Image folder or a video file")
    ask.add_argument("query", type=str, help='Question, for example "where is the blue water bottle?"')
    ask.add_argument(
        "--sweep",
        nargs=3,
        type=float,
        metavar=("START_PAN", "END_PAN", "TILT"),
    )
    ask.add_argument("--angles-csv", type=Path)
    ask.add_argument("--out-dir", type=Path, default=None)
    ask.add_argument("--no-cache", action="store_true")
    ask.add_argument(
        "--debug",
        action="store_true",
        help="Write annotated images under the output debug folder",
    )

    answer_cmd = sub.add_parser(
        "answer",
        help="Index a scan and print fire_laser, aim, the query result, and the item list.",
    )
    answer_cmd.add_argument("scan_dir", type=Path, help="Image folder or a video file")
    answer_cmd.add_argument("query", type=str, help='Question, for example "where is the blue water bottle?"')
    answer_cmd.add_argument(
        "--sweep",
        nargs=3,
        type=float,
        metavar=("START_PAN", "END_PAN", "TILT"),
    )
    answer_cmd.add_argument("--angles-csv", type=Path)
    answer_cmd.add_argument("--out-dir", type=Path, default=None)
    answer_cmd.add_argument("--no-cache", action="store_true")
    answer_cmd.add_argument(
        "--debug",
        action="store_true",
        help="Write annotated images under the output debug folder",
    )

    evaluate = sub.add_parser(
        "eval",
        help="Score a scene's ground_truth.json. Pass a directory or a name under data/test_scenes.",
    )
    evaluate.add_argument("scene", type=str)
    evaluate.add_argument("--out-dir", type=Path, default=None)
    evaluate.add_argument("--no-cache", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except Exception as exc:
        print(redact(str(exc)), file=sys.stderr)
        return 1

    out_dir = getattr(args, "out_dir", None)
    if out_dir is not None:
        settings = settings.model_copy(update={"out_dir": str(out_dir)})

    try:
        if args.command == "check-config":
            return _check_config(settings)
        if args.command == "describe":
            return _describe(settings, args.image, args.out_dir, use_cache=not args.no_cache)
        if args.command == "index":
            return _index(settings, args)
        if args.command == "ask":
            return _ask(settings, args)
        if args.command == "answer":
            return _answer(settings, args)
        if args.command == "eval":
            return _eval_scene(settings, args)
    except MissingAPIKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (GrokCallError, IngestError, EvalError) as exc:
        print(redact(str(exc)), file=sys.stderr)
        return 1
    parser.error(f"Unknown command {args.command}")
    return 2


def _check_config(settings) -> int:
    key_state = "present" if api_key_is_present(settings) else "missing"
    lines = [
        f"provider: {settings.provider}",
        f"fast_model: {settings.fast_model}",
        f"reasoning_model: {settings.reasoning_model}",
        f"base_url: {settings.base_url}",
        f"api_key_env: {settings.api_key_env}",
        f"{settings.api_key_env}: {key_state}",
        f"image_detail: {settings.image_detail}",
        f"query_model: {settings.active_query_model}",
        f"tile_grid: {list(settings.tile_grid)}",
        f"max_concurrency: {settings.max_concurrency}",
        f"api_max_edge: {settings.api_max_edge}",
        f"verify_match: {settings.verify_match}",
        f"save_debug: {settings.save_debug}",
        f"request_timeout_s: {settings.request_timeout_s}",
        f"cache_dir: {settings.cache_path}",
        f"out_dir: {settings.out_path}",
        f"detector: {settings.detector}",
    ]
    print("\n".join(lines))
    return 0


def _video_kwargs(settings, args) -> dict:
    sweep = tuple(args.sweep) if getattr(args, "sweep", None) else None
    return {
        "sweep": sweep,
        "angles_csv": getattr(args, "angles_csv", None),
        "use_cache": not args.no_cache,
        "save_debug": bool(settings.save_debug or getattr(args, "debug", False)),
    }


def _index(settings, args) -> int:
    require_api_key(settings)
    catalog = build_catalog(args.scan_dir, settings=settings, **_video_kwargs(settings, args))
    print(catalog.model_dump_json(indent=2))
    print(f"wrote {settings.out_path / 'catalog.json'}", file=sys.stderr)
    return 0


def _ask(settings, args) -> int:
    require_api_key(settings)
    catalog = catalog_for_query(args.scan_dir, settings=settings, **_video_kwargs(settings, args))
    result = locate(args.query, catalog, settings=settings, use_cache=not args.no_cache)
    print(json.dumps(public_query_dict(result), indent=2))
    return 0


def _answer(settings, args) -> int:
    require_api_key(settings)
    payload = answer(args.scan_dir, args.query, settings=settings, **_video_kwargs(settings, args))
    print(json.dumps(payload, indent=2))
    return 0


def _eval_scene(settings, args) -> int:
    require_api_key(settings)
    report = evaluate_scene(args.scene, settings=settings, use_cache=not args.no_cache)
    print(report.model_dump_json(indent=2))
    return 0


def _describe(settings, image: Path, out_dir: Path | None, *, use_cache: bool) -> int:
    destination_dir = out_dir if out_dir is not None else settings.out_path
    try:
        result, cache_state = describe_image(image, settings, use_cache=use_cache)
    except MissingAPIKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except GrokCallError as exc:
        print(redact(str(exc)), file=sys.stderr)
        return 1

    destination_dir.mkdir(parents=True, exist_ok=True)
    json_path = destination_dir / f"{image.stem}.json"
    image_path = destination_dir / f"{image.stem}_annotated.png"
    payload = result.model_dump()
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    annotate_image(image, result.objects, image_path, legend=legend_name(settings.provider))

    print(json.dumps(payload, indent=2))
    print(f"cache: {cache_state}", file=sys.stderr)
    print(f"wrote {json_path}", file=sys.stderr)
    print(f"wrote {image_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
