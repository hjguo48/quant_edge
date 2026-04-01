from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import scripts.fetch_data as fetch_data_script


def test_fetch_data_parse_args_defaults() -> None:
    args = fetch_data_script._parse_args([])

    assert args.start_date == "2018-01-01"
    assert args.end_date is None
    assert args.polygon_request_interval == 12.5
    assert args.fmp_request_interval == 0.5


def test_fetch_data_help_works_without_runtime_imports() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "fetch_data.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--start-date" in result.stdout
    assert "--ticker-start" in result.stdout
    assert "--polygon-request-interval" in result.stdout
