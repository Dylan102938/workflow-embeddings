import argparse
import importlib
from pathlib import Path

from ...storage import paths
from .common import download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=paths.ADDED_DATASETS)
    parser.add_argument("--raw", help="the downloaded dataset (default: DATA/added/sources/<dataset>)")
    parser.add_argument("--download", action="store_true", help="first download what the converter reads")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    converter = importlib.import_module(f"{__package__}.{args.dataset}")
    raw_directory = Path(args.raw) if args.raw else paths.SOURCES / args.dataset
    if args.download:
        for patterns in converter.download_patterns(raw_directory):
            download(converter.REPOSITORY, patterns, raw_directory)
    converter.convert(raw_directory, args.workers)


if __name__ == "__main__":
    main()
