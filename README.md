# momentum-system

Autonomous weekly momentum stock ranking system.

## Structure

```
momentum-system/
├── .github/workflows/
│   ├── weekly_pipeline.yml     # Friday 6pm ET — full pipeline run
│   └── monday_scoring.yml      # Monday 7am ET — score actuals + self-refine
├── pipeline/
│   ├── 01_universe.py          # NYSE+NASDAQ liquidity filter → data/universe.csv
│   ├── 02_regime.py            # Market regime classifier → data/regime.json
│   ├── 03_signals.py           # Signal engine → data/signals.csv
│   ├── 04_model.py             # LightGBM scoring → data/scores.csv
│   ├── 05_llm_synthesis.py     # Claude API top-50 thesis → data/synthesis.json
│   └── 06_report.py            # HTML report → reports/YYYY-MM-DD.html
├── config/
│   └── weights.json            # Signal weights — only thing Claude can modify
├── data/
│   ├── universe.csv            # Current filtered universe
│   ├── regime.json             # Current regime state
│   ├── signals.csv             # All signal scores per ticker
│   ├── scores.csv              # Final ranked scores
│   └── performance_log.csv     # Score vs actual return — grows weekly
├── models/
│   └── lgbm_model.pkl          # Trained model (retrained periodically)
├── refinements/
│   └── YYYY-MM-DD.md           # Claude's weekly reasoning for weight changes
├── reports/
│   └── YYYY-MM-DD.html         # Weekly report archive
├── run_pipeline.py             # Orchestrator
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run manually

```bash
python run_pipeline.py
```

## Automation

Two GitHub Actions workflows handle everything automatically.
See `.github/workflows/` for cron schedules and secrets required.
Required secrets: `ANTHROPIC_API_KEY`
