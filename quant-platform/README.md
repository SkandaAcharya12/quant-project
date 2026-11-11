# Algorithmic Trading & Portfolio Optimization Platform

Four isolated engines plus a dashboard:

| Module | What it does |
|---|---|
| `data_engine/` | Thread-safe Yahoo Finance ingestion, gap handling, macro indicators, from-scratch technical indicators |
| `backtester/` | Cointegration-gated pairs trading: rolling Engle-Granger test, Z-score signals, delayed fills, costs |
| `optimizer/` | Mean-variance optimization with Ledoit-Wolf / Bayes-Stein shrinkage, efficient frontier, 10,000-run Monte Carlo, walk-forward test |
| `risk_engine/` | Historical / parametric / Cornish-Fisher / filtered-historical VaR and ES, drawdowns, Kupiec and Christoffersen backtests, stress tests |
| `app.py` | Streamlit dashboard over all four engines |

Everything is chosen or estimated on a **formation period** (default through 2024-12-31) and judged on a later
**out-of-sample period** (default from 2025-01-01).

## Testing and running (exact commands)

Run these from inside the `quant-platform` folder. The only pip command you need is the one that installs
`requirements-dev.txt`; it pulls in everything.

### First time (or after changing Python versions)

Windows (Command Prompt):
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```
Mac / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Every time you receive an update
1. Replace the changed files (or re-extract the zip), open a terminal in the folder, and activate the environment
   (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Mac/Linux).
2. If `requirements.txt` changed (the update notes will say so): `pip install -r requirements-dev.txt`
3. Run the test suite (no internet needed):
   ```
   python -m pytest
   ```
4. Run the live demos (needs internet):
   ```
   python main.py                      # everything
   python main.py --only data          # data engine
   python main.py --only pairs         # pairs trading
   python main.py --only optimizer     # portfolio optimizer
   python main.py --only risk          # risk engine
   ```
5. Launch the dashboard (opens http://localhost:8501; stop it with Ctrl+C):
   ```
   streamlit run app.py
   ```
6. Optional code-quality checks (strict typing and lint; may report issues worth sending back):
   ```
   mypy data_engine backtester optimizer risk_engine
   ruff check .
   ```

### Changing the out-of-sample dates
```
python main.py --formation-end 2024-12-31 --oos-start 2025-01-01
```

### What `pip install -r requirements-dev.txt` installs
numpy, pandas, scipy, statsmodels, scikit-learn, yfinance, fredapi, pyarrow, requests, pydantic, python-dotenv,
plotly, matplotlib, streamlit, pytest, pytest-cov, hypothesis, mypy, pandas-stubs, ruff.
If one package fails, install it alone, e.g. `pip install statsmodels`, and rerun the full command.

## Notes
* Optional FRED macro data: copy `.env.example` to `.env` and add a free `FRED_API_KEY`.
* Every formula is derived in the docstring of the module that uses it.
* Educational research tool, not investment advice.
