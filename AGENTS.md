# Codex Agent Instructions

##  Dove lavorare
- Codice sorgente principale: `trading_backtest/`
- Strategie personalizzate: `trading_backtest/strategy/`
- Script di esecuzione: `run.py` (entry point del progetto)
- Ottimizzazione con Optuna: `trading_backtest/optimize.py`
- Calcolo delle metriche: `trading_backtest/performance.py`
- Test automatizzati: `tests/` (se presenti)

##  Come eseguire il progetto
- Avvia l’intero processo:  
  python run.py

- Dati attesi: file CSV con OHLC BTC 15min (configurabile in `config.py`)

##  Validazione dei cambiamenti
- Linter:  
  black trading_backtest/

- Test (se presenti):  
  pytest

- Suggerito: aggiungi test per ogni nuovo modulo/funzione

##  Obiettivi per Codex
- Migliorare la logica delle strategie (`strategy/*.py`)
- Ottimizzare parametri tramite Optuna (`optimize.py`)
- Migliorare il calcolo delle metriche (`performance.py`)
- Aggiungere e mantenere test automatici (`tests/`)
- *(Facoltativo)* Esporre `run.py` come CLI con argomenti (`argparse` o `typer`)

##  Ambiente consigliato
- Python 3.x
- Dipendenze:  
  pip install -r requirements.txt

- Tool opzionali:  
  pip install black pytest

##  Linee guida PR
- Titolo in formato: `[modulo] Descrizione sintetica`
- Messaggio PR in inglese, chiaro e descrittivo
- Codice formattato e testato
- Evita modifiche non correlate nella stessa PR

