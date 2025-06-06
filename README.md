# Trading Backtest Project

Questo repository contiene un framework leggero per testare diverse strategie di trading su serie storiche di prezzo. I dati per il backtest sono letti da un file CSV, per impostazione predefinita `data/btc_15m_data_from_2021.csv` (variabile d'ambiente `DATA_FILE`).

## Mappa del progetto

```
trading_backtest_project/
├── run.py
├── AGENTS.md
├── requirements.txt
├── setup.sh
├── trading_backtest/
│   ├── __main__.py
│   ├── benchmark.py
│   ├── config.py
│   ├── data.py
│   ├── optimize.py
│   ├── performance.py
│   ├── strategy/
│   │   ├── base.py
│   │   ├── sma.py
│   │   ├── rsi.py
│   │   ├── breakout.py
│   │   ├── bollinger.py
│   │   ├── momentum.py
│   │   ├── macd.py
│   │   ├── stochastic.py
│   │   └── random_forest.py
│   └── utils/
│       └── io_utils.py
└── tests/
    └── ...
```

### Responsabilità dei moduli principali

- **`run.py`**: semplice entrypoint che richiama `trading_backtest.__main__.main()`.
- **`trading_backtest/__main__.py`**: gestisce la CLI (`--strategy`, `--trials`, `--benchmark`) e coordina caricamento dati, calcolo indicatori e ottimizzazione.
- **`config.py`**: definisce percorsi, logging e dataclass con i parametri per ogni strategia.
- **`data.py`**: funzioni per caricare il CSV e aggiungere al DataFrame gli indicatori tecnici utilizzati dalle strategie.
- **`performance.py`**: classe `PerformanceAnalyzer` per metriche come total return, Sharpe ratio e drawdown.
- **`optimize.py`**: definisce gli spazi di ricerca per Optuna e funzioni di valutazione/pruning delle strategie.
- **`benchmark.py`**: lancia l'ottimizzazione di ciascuna strategia e produce un riepilogo dei risultati.
- **`strategy/`**: contiene la classe astratta `BaseStrategy` e le varie implementazioni (SMA, RSI, Breakout, Bollinger, Momentum, MACD, Stochastic, RandomForest).
- **`utils/io_utils.py`**: funzioni di supporto per la lettura/scrittura di CSV.

## Setup

Assicurarsi di avere tutte le dipendenze installate:

```bash
pip install -r requirements.txt
# oppure
./setup.sh
```

## Esecuzione

Posizionare il file CSV dei prezzi nel percorso indicato (o impostare `DATA_FILE`). Esempio di avvio per ottimizzare la strategia SMA:

```bash
python run.py --strategy sma --trials 100
```

Per eseguire un benchmark veloce di tutte le strategie:

```bash
python run.py --benchmark
```

Impostando la variabile `RUN_ML=1` viene inclusa anche la strategia RandomForest.

## Sviluppo

Prima di aprire una pull request formatta il codice con `black` e assicurati che i test passino:

```bash
black trading_backtest tests
pytest
```

La verbosità dei log è regolabile tramite l'ambiente `LOG_LEVEL`.
