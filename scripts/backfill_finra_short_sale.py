#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill FINRA daily short-sale volume files into short_sale_volume_daily.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--markets",
        default="CNMS,ADF,BNY",
        help="Comma-separated FINRA market prefixes.",
    )
    parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Ignore cached ETags and re-download files.",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def parse_date(raw: str) -> date:
    return date.fromisoformat(raw)


def parse_markets(raw: str) -> list[str]:
    return [segment.strip().upper() for segment in raw.split(",") if segment.strip()]


def main(argv: list[str] | None = None) -> int:
    from src.data.finra_short_sale import FINRAShortSaleSource

    configure_logging()
    args = parse_args(argv)
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    markets = parse_markets(args.markets)
    source = FINRAShortSaleSource(min_request_interval=0.0)
    inserted = source.fetch_historical(
        start_date=start_date,
        end_date=end_date,
        markets=markets,
        force_refetch=args.force_refetch,
    )
    logger.info(
        "FINRA short-sale backfill finished: start={} end={} markets={} rows={}",
        start_date,
        end_date,
        markets,
        inserted,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
