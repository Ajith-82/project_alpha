import argparse
import pickle
from classes.DatabaseManager import (
    connect_db,
    create_tables,
    insert_price_rows,
    insert_company_info,
)


def migrate(pickle_file: str, db_path: str) -> None:
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    conn = connect_db(db_path)
    create_tables(conn)

    for symbol, df in data.get("price_data", {}).items():
        insert_price_rows(conn, symbol, df)

    for symbol, info in data.get("company_info", {}).items():
        insert_company_info(conn, symbol, info)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate pickle cache to SQLite")
    parser.add_argument("pickle_file", help="Existing pickle data file")
    parser.add_argument("db_path", help="SQLite database path")
    args = parser.parse_args()
    migrate(args.pickle_file, args.db_path)

