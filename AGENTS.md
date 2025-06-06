# Contributor Guidelines

This project uses Python. When updating the codebase, please:

1. Install dependencies with `pip install -r requirements.txt` or run `./setup.sh`.
2. Format any Python files with `black` before committing.
3. Run the unit tests with `pytest`.
4. The main entry points are `run.py` or `python -m trading_backtest`.
5. Price data is read from the CSV path given by the `DATA_FILE` environment variable. If unset, it defaults to `data/btc_15m_data_from_2021.csv`.
