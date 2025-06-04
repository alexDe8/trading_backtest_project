# Trading Backtest Project

## Data path

The project expects a CSV file with price data. By default the path is
`data/btc_15m_data_2018_to_2025.csv` relative to the project root. You can use a
different location by setting the environment variable `DATA_FILE` before running
the program:

```bash
export DATA_FILE=/path/to/your/data.csv
python run.py
```

Alternatively, place the CSV file at the default location and simply run
`python run.py`.
