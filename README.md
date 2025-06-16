# Project Alpha

Project Alpha is a collection of Python tools for scanning and analysing
stock markets.  It fetches historical prices, trains a volatility model, and
runs a suite of technical screeners.  The project can work with both US and
Indian equities and stores data either as pickle files or in a SQLite
cache.

## Installation

```bash
pip install --user poetry
poetry install
```

Poetry creates an isolated virtual environment for the project. Activate it with:

```bash
poetry shell
```

## Usage

The main entry point is `src/project_alpha.py`.  A few important command-line
options are shown below:

```text
--market      Market name to fetch stocks list ("us" or "india")
--save-table  Save prediction table in csv format
--no-plots    Plot estimates with their uncertainty over time
--db-path     Path to SQLite database for storing price data
```

Run the script directly for US stocks:

```bash
python src/project_alpha.py
```

Or specify the Indian market:

```bash
python src/project_alpha.py --market india
```

Logged executions are provided via helper shell scripts
`run_us_stock_scanner.sh` and `run_india_stock_scanner.sh`.

## Database Caching

Price data can be saved in an SQLite file using `--db-path`.  The database
file will be created automatically along with any missing parent directories.
The `migrate_pickle_to_db.py` script converts existing pickle caches to the new
format.

## Output

Processed charts and prediction tables are written to the `data/processed_data`
folder.  Email functions are included for sending results, but credentials must
be supplied separately.

This repository contains a number of Jupyter notebooks for exploratory work and
model experimentation.  They are optional for running the basic pipeline.

Project Alpha is provided for educational purposes only and does not constitute
financial advice.
