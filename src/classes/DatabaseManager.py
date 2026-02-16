import sqlite3
import pandas as pd
import json
from typing import Optional
import os


def connect_db(db_path: str) -> sqlite3.Connection:
    """Return a connection to the SQLite database at ``db_path``.

    Any missing parent directories are created automatically so that a new
    database file can be initialised without manual setup.
    """
    parent = os.path.dirname(db_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return sqlite3.connect(db_path)


def create_tables(conn: sqlite3.Connection) -> None:
    """Create required tables if they do not already exist."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS price_data(
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume REAL,
            dividends REAL,
            splits REAL,
            PRIMARY KEY(symbol, date)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS company_info(
            symbol TEXT PRIMARY KEY,
            info_json TEXT
        )
        """
    )
    conn.commit()


def get_last_date(conn: sqlite3.Connection, symbol: str) -> Optional[str]:
    """Return the latest date stored for ``symbol`` or ``None``."""
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM price_data WHERE symbol=?", (symbol,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def insert_price_rows(conn: sqlite3.Connection, symbol: str, df: pd.DataFrame) -> None:
    """Insert or replace rows from ``df`` into ``price_data``."""
    if df is None or df.empty:
        return
    df = df.copy()
    if df.index.name is None:
        df.index.name = "date"
    df.reset_index(inplace=True)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    
    # Ensure date is string for sqlite
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
        
    df["symbol"] = symbol
    records = df[
        [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "dividends",
            "stock_splits" if "stock_splits" in df.columns else "splits",
        ]
    ].values.tolist()
    sql = (
        "INSERT OR REPLACE INTO price_data "
        "(symbol, date, open, high, low, close, adj_close, volume, dividends, splits) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    conn.executemany(sql, records)
    conn.commit()


def insert_company_info(conn: sqlite3.Connection, symbol: str, info: dict) -> None:
    """Insert or replace company info."""
    if not info:
        return
    data = json.dumps(info)
    conn.execute(
        "INSERT OR REPLACE INTO company_info(symbol, info_json) VALUES(?, ?)",
        (symbol, data),
    )
    conn.commit()


def get_price_dataframe(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    """Return a DataFrame of historical prices for ``symbol``."""
    query = (
        "SELECT date, open, high, low, close, adj_close, volume, dividends, splits "
        "FROM price_data WHERE symbol=? ORDER BY date"
    )
    df = pd.read_sql_query(query, conn, params=(symbol,))
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
            "dividends": "Dividends",
            "splits": "Stock Splits",
        },
        inplace=True,
    )
    return df


def get_company_info(conn: sqlite3.Connection, symbol: str) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute("SELECT info_json FROM company_info WHERE symbol=?", (symbol,))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None

