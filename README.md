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

A small synthetic dataset is provided in the `data/` directory for testing
purposes. The CSV must include the following columns:

| column     | description                            |
|----------- |----------------------------------------|
| `Open time`| timestamp of the 15 minute period       |
| `Open`     | opening price                          |
| `High`     | highest price within the period        |
| `Low`      | lowest price within the period         |
| `Close`    | closing price                          |
| `Volume`   | traded volume                          |

Column names need to match exactly; any extra columns are ignored. The loader
converts `Open time` into a `timestamp` column internally.

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
from trading_backtest.strategy import get_strategy
from trading_backtest.config import SMAConfig

cfg = SMAConfig(
    sma_fast=10,
    sma_slow=50,
    sma_trend=200,
    sl_pct=5,
    tp_pct=20,
    position_size=0.1,
    trailing_stop_pct=2.0,
)
strategy_cls, _ = get_strategy("sma")
strat = strategy_cls(cfg)
trades = strat.generate_trades(df)
```

For strategies without the `position_size` parameter, `qty` defaults to `1`.

## Command line option

The main entry point supports selecting which strategy to optimize. Use
`--strategy` or set the `STRATEGY` environment variable. Available values are
`sma`, `rsi`, `breakout`, `bollinger`, `momentum` and `vol_expansion`. When the
option is omitted, `sma` is used by default:

```bash
python run.py --strategy rsi
# or via environment variable
STRATEGY=breakout python run.py
```

