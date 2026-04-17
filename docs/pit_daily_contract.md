# Daily PIT Contract

This document defines the point-in-time contract for daily strategies in QuantEdge.

## Execution Model

- Signal formation: compute scores after the market close on trade date `T`.
- Execution: submit the trade for execution at the next session open, `T+1 open`.
- Daily labels for short horizons must therefore use `T+1 open` as the entry price.

This contract is stricter than "same-day close execution" and is consistent with the repo's PIT storage model for historical daily bars.

## Daily Bar Knowledge Time

### Rule

- A daily `OHLCV` bar for trade date `T` is not treated as knowable on `T`.
- Historical daily bars become visible at `T+1 16:00 ET` in the Polygon ingestion path.
- In practical research terms, the `T` bar is only available for a signal generated after the close and applied from `T+1` onward.

### Repo Verification

- `src/data/sources/polygon.py`
  - `_historical_knowledge_time(trade_date)` sets daily bar `knowledge_time` to `trade_date + 1 day, 16:00 ET`.
- `src/data/db/pit.py`
  - `get_prices_pit()` enforces `StockPrice.knowledge_time <= as_of`.
- `tests/test_data/test_pit.py`
  - `test_prices_pit_respects_knowledge_time()` verifies rows are excluded until their `knowledge_time` arrives.

## Existing Pipeline Assumption

The current feature pipeline already matches this contract.

- `src/features/pipeline.py`
  - `FeaturePipeline.run()` documents that returned `trade_date` values remain market dates while source prices may only become knowable after the market date.
  - It explicitly warns that if `as_of == end_date`, the last market date may be unavailable when prices use `T+1 knowledge_time`.

Conclusion: the pipeline already assumes `daily bar visible on T+1`, so the new short-horizon labels must align with that same timing.

## Label Contract For Daily Strategies

For `1D` and `5D` labels:

- Entry price: `open[T+1]`
- Exit price:
  - `1D`: `open[T+2]`
  - `5D`: `open[T+6]`
- Benchmark return must use the same open-to-open window.

This is now implemented in `src/labels/forward_returns.py` for horizons `1` and `5`.

Longer horizons such as `20D` and `60D` currently retain the existing close-to-close convention so current medium-horizon research artifacts remain stable.

## PIT Rules By Data Type

| Data type | PIT availability rule | Repo / contract source |
| --- | --- | --- |
| Daily price (`OHLCV`) | `T+1` knowledge time | `src/data/sources/polygon.py`, `src/data/db/pit.py` |
| Fundamentals | `acceptedDate` / `fillingDate` / filing-driven `knowledge_time` | `src/data/sources/fmp.py`, `src/data/db/pit.py` |
| Analyst / consensus | publication calendar date becomes `knowledge_time` at market close | `src/data/sources/fmp.py` earnings consensus path |
| Short interest | treat as `T+2` availability by contract | Stage 1 research contract; not yet implemented in current ingestion code |

## Practical Research Implications

- Do not backtest a daily strategy as if `close[T]` were tradeable at the same timestamp used to compute the score.
- Use `as_of` timestamps that are strictly after the relevant `knowledge_time`.
- Keep purge/embargo logic in the walk-forward layer, not in label generation.

## Summary

- Daily bars are PIT-safe only after `knowledge_time <= as_of`.
- Historical daily price knowledge is `T+1`.
- Daily strategy execution is `close[T] signal -> open[T+1] execution`.
- Short-horizon labels must therefore be open-to-open, not close-to-close.
