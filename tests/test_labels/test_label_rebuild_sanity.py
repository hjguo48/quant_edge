from __future__ import annotations

from datetime import date

import pandas as pd

from scripts.rebuild_forward_labels import annotate_label_validity


def test_annotate_label_validity_marks_pre_spy_rows_invalid() -> None:
    labels = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade_date": date(2016, 4, 15),
                "horizon": 10,
                "forward_return": 0.05,
                "excess_return": pd.NA,
            },
            {
                "ticker": "AAA",
                "trade_date": date(2016, 4, 20),
                "horizon": 10,
                "forward_return": pd.NA,
                "excess_return": pd.NA,
            },
            {
                "ticker": "AAA",
                "trade_date": date(2016, 4, 21),
                "horizon": 10,
                "forward_return": 0.04,
                "excess_return": 0.01,
            },
        ],
    )

    annotated = annotate_label_validity(labels, benchmark_start=date(2016, 4, 18))

    first = annotated.iloc[0]
    second = annotated.iloc[1]
    third = annotated.iloc[2]

    assert bool(first["is_valid_excess"]) is False
    assert first["invalid_reason"] == "benchmark_unavailable_pre_spy"
    assert bool(second["is_valid_excess"]) is False
    assert second["invalid_reason"] == "insufficient_forward_window"
    assert bool(third["is_valid_excess"]) is True
    assert pd.isna(third["invalid_reason"])
