# Trading Backtest Project

## Data path

The project expects a CSV file with price data. If the environment variable
`DATA_FILE` is not set, the default path is
`data/btc_15m_data_2018_to_2025.csv` relative to the project root. You can use a
different location by setting `DATA_FILE` before running the program:

```bash
export DATA_FILE=/path/to/your/data.csv
python run.py
```

Alternatively, place the CSV file at the default location and simply run
`python run.py`.

## Setup

Before running the project or the test suite make sure all dependencies are
installed:

```bash
pip install -r requirements.txt
# or use the helper script
./setup.sh
```

## Strategy usage

Each strategy exposes a `generate_trades(df)` method. The SMA crossover strategy
accepts an additional `position_size` parameter used to size trades. The
generated DataFrame now includes a `qty` column with this value:

```python
from trading_backtest.strategy.sma import SMACrossoverStrategy

strat = SMACrossoverStrategy(
    sma_fast=10,
    sma_slow=50,
    sma_trend=200,
    sl_pct=5,
    tp_pct=20,
    position_size=0.1,
    trailing_stop_pct=2.0,
)
trades = strat.generate_trades(df)
```

For strategies without the `position_size` parameter, `qty` defaults to `1`.

## Command line interface

The package exposes a small CLI. You can run the optimiser directly using the
module or `run.py` and configure data path, strategy and number of trials:

```bash
# optimise the RSI strategy using custom data
python -m trading_backtest --data /path/to/data.csv --strategy rsi --trials 100

# equivalent invocation via the helper script
python run.py --strategy breakout --trials 200
```

Running the module without arguments defaults to the SMA strategy and uses the
`DATA_FILE` environment variable when set.

