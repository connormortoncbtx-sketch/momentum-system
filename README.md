# Position Sizing Patch — 10×10% baseline

Implements the canonical deployment philosophy: 10 positions × 10% of capital.

## Philosophy (encoded in code comments)

Baseline deployment is 10 positions × 10% of capital. This is the canonical
shape; everything else is a variation. Position COUNT is the primary lever,
position SIZE is derived. `MIN_POSITIONS = 10` is a hard floor — below 10
qualifying names, deploy what's available and alert.

Future work (post-paper-trading, not in this patch): dynamic position count
growth beyond 10 when per-ticker liquidity constraints force dilution. The
threshold data needed to calibrate this comes from paper trading slippage
and ADV observations.

## What changed

File: `automation/alpaca_trader.py`

### Constants section

Added:
- `DEFAULT_POSITIONS = 10` — target count
- `MIN_POSITIONS = 10` — hard floor with alert, not a refusal

Retained (unchanged):
- `MIN_POSITION_SIZE = 500.0`
- `MAX_POSITION_PCT = 0.15`
- `MIN_COMPOSITE_PERCENTILE = 0.70`
- `MAX_ADV_PCT = 0.01`
- `MIN_PRICE = 1.00`

### compute_positions() function

Replaced:
```python
max_by_capital = int(portfolio_value / MIN_POSITION_SIZE)
# ... liquidity filter using est_position = portfolio_value / min(...) ...
n_positions = min(max_by_capital, len(df))
```

With:
```python
max_by_capital = int(portfolio_value / MIN_POSITION_SIZE)
# ... liquidity filter using target_position = portfolio_value / DEFAULT_POSITIONS ...
n_positions = min(max_by_capital, len(df), DEFAULT_POSITIONS)
# ... report which of the three constraints bound the count ...
# ... if n_positions < MIN_POSITIONS: warn + notify ...
```

Plus new logging: every entry now prints which constraint bound the position
count (transaction cost floor / qualifying pool / default target). This
surfaces design signal every week.

Plus new notification: if n_positions < MIN_POSITIONS, the Tier 3 notifier
sends a HIGH priority alert so you can investigate why the qualifying pool
was narrow.

## Behavior across capital levels

| Capital | tx floor | n_positions | per-position | % of portfolio | bound by |
|---|---|---|---|---|---|
| $1,000 | 2 | 2 | $150* | 15% | transaction cost floor ⚠ below MIN |
| $3,000 | 6 | 6 | $450* | 15% | transaction cost floor ⚠ below MIN |
| $5,000 | 10 | 10 | $500 | 10% | default target |
| $7,500 | 15 | 10 | $750 | 10% | default target |
| $25,000 | 50 | 10 | $2,500 | 10% | default target |
| $100,000 | 200 | 10 | $10,000 | 10% | default target |
| $1,000,000 | 2,000 | 10 | $100,000 | 10% | default target |

*Binds against MAX_POSITION_PCT = 15% at very small capital. Position count
below 10 triggers the notification.

## Key behavioral changes vs. prior code

**At $7,500 (current Fidelity)**: was 15 positions × $500, now 10 positions
× $750. Matches what you actually do manually.

**At $100k (upcoming Alpaca paper)**: was 200 positions × $500, now 10
positions × $10,000. Matches your philosophy of concentrating in highest-
conviction names rather than diluting down the rank order.

**At any scale**: position count no longer grows with capital. Size grows.
When position count needs to grow (future work), it will be triggered by
per-ticker liquidity constraints, not by capital/floor arithmetic.

## Low-qualifying-pool behavior

If fewer than 10 names qualify for entry (after composite percentile filter,
liquidity filter, price filter, etc.):

1. Deploy what's available (don't refuse to trade)
2. Log a warning with the actual count
3. Send a HIGH priority notification via the Tier 3 observability layer
4. Each deployed position still sizes to its fair share (e.g., 8 qualifying
   names = 8 positions × 12.5% each, still under MAX_POSITION_PCT=15%)

This is "Option 2" from the discussion — practical floor with visibility,
not an absolute refusal to trade below the floor.

## Deploy

Drop in place. No data migration needed. Monday's 2:30pm CT entry will use
the new sizing logic automatically.

## Verification

- Syntax: `py_compile` passes
- Simulation: correct behavior at $1k, $3k, $5k, $7.5k, $10k, $25k, $100k,
  $500k, $1M, $10M (see table above)
- MIN_POSITIONS alert fires correctly below $5k
- Default target binds correctly at $5k and above
- Position sizing honors MAX_POSITION_PCT at very small capital

## What's explicitly NOT in this patch

- Dynamic position count growth based on per-ticker liquidity caps
- VWAP/TWAP execution routing when individual position size exceeds
  impact thresholds
- Universe floor ratcheting (raising MIN_MARKET_CAP as capital grows)

All three are planned post-paper-trading features that require empirical
calibration data this system doesn't yet have. Leaving them unbuilt now is
intentional.
