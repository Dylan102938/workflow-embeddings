import argparse
import importlib
from pathlib import Path

from ...storage import paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=paths.ADDED_DATASETS)
    parser.add_argument("--raw", help="the downloaded dataset (default: DATA/sources/<dataset>)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    converter = importlib.import_module(f"{__package__}.{args.dataset}")
    converter.convert(Path(args.raw) if args.raw else paths.SOURCES / args.dataset, args.workers)


if __name__ == "__main__":
    main()
