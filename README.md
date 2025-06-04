# Trading Backtest Project

This repository provides a minimal pipeline to backtest and optimize
cryptocurrency trading strategies. The main script loads historic price
data, caches indicators and searches for profitable parameters using
[Optuna](https://optuna.org/). Results are written to CSV files to allow
further analysis.

## Setup

1. Install the project dependencies:

```bash
pip install -r requirements.txt
pip install black pytest  # optional tools
```

The `setup.sh` script performs the same installation steps.

## Running the pipeline

1. Place your BTC 15 minute OHLC data in CSV format somewhere on your
   machine. Update the path in `trading_backtest/__main__.py` (or change
   `DATA_FILE` in `trading_backtest/config.py`) so that
   `load_price_data()` can read your file.
2. Execute the backtest:

```bash
python run.py
```

The script optimizes parameters for the SMA crossover strategy using
Optuna, performs a grid search around the best trial and compares a few
reference strategies. The aggregated results are saved to
`results_live.csv` and `summary_live.csv`.

## Optional tests and style checks

- Run `pytest` to execute any available unit tests.
- Run `black trading_backtest/` to apply the repository's formatting
  conventions.

