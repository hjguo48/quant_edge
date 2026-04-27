#!/usr/bin/env python3
"""Freeze the active S&P 500 ticker universe at a given as-of date and dump
to JSON. Used to keep ticker membership identical across multiple
``run_per_horizon_ic_screening.py`` runs (e.g. when chunking by date).

Without this, each chunk's ``build_sample_context`` calls
``get_historical_members(end_date, ...)`` and the resulting 503-name list can
drift between chunks, breaking exact equivalence with a single full-span run
(data audit 2026-04-25 follow-up: chunked IC merge plan).

Usage:
    python scripts/freeze_universe.py --as-of 2025-02-28 \
        --output data/features/frozen_universe_503.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.universe.history import get_historical_members  # noqa: E402


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=_parse_date, required=True,
                        help="As-of date for active SP500 membership snapshot.")
    parser.add_argument("--index-name", default="SP500")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path.")
    args = parser.parse_args()

    tickers = sorted(set(get_historical_members(args.as_of, args.index_name)))
    if not tickers:
        print(f"[freeze_universe] WARNING: no tickers returned for {args.as_of}")
        return 1

    payload = {
        "as_of": args.as_of.isoformat(),
        "index_name": args.index_name,
        "ticker_count": len(tickers),
        "tickers": tickers,
        "frozen_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[freeze_universe] wrote {args.output} with {len(tickers)} tickers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
