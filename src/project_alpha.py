#!/usr/bin/env python
import sys
import datetime as dt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
from json import tool
import os.path
import pickle

from classes.Download import download
from classes.Volatile import volatile
from classes.Screener_one import screener_one
import classes.IndexListFetcher as Index
import classes.Tools as tools


def cli_argparser():
    cli = ArgumentParser(
        "Volatile: your day-to-day trading companion.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    cli.add_argument("-s", "--symbols", type=str, nargs="+", help=SUPPRESS)
    cli.add_argument(
        "--rank",
        type=str,
        default="rate",
        choices=["rate", "growth", "volatility"],
        help="If `rate`, stocks are ranked in the prediction table and in the stock estimation plot from "
        "the highest below to the highest above trend; if `growth`, ranking is done from the largest"
        " to the smallest trend growth at current date; if `volatility`, from the largest to the "
        "smallest current volatility estimate.",
    )
    cli.add_argument(
        "--market",
        type=str,
        default="us",
        choices=["us", "india"],
        help="Market name to fetch stocks list",
    )
    cli.add_argument(
        "--save-table", action="store_true", help="Save prediction table in csv format."
    )
    cli.add_argument(
        "--no-plots",
        action="store_true",
        help="Plot estimates with their uncertainty over time.",
    )
    cli.add_argument(
        "--plot-losses",
        action="store_true",
        help="Plot loss function decay over training iterations.",
    )
    cli.add_argument(
        "--cache",
        action="store_true",
        help="Use cached data and parameters if available.",
    )
    return cli.parse_args()

def load_data(cache, symbols=None, market="", file_prefix="", data_dir=""):
    while cache:
        print("\nLoading historical data...")
        data = tools.load_cache(file_prefix, data_dir)
        if bool(data):
            return data
        else:
            cache = ""

    if symbols is None:
        symbols_file_path = "symbols_list.txt"
        if os.path.exists(symbols_file_path):
            with open(symbols_file_path, "r") as my_file:
                symbols = my_file.readline().split(" ")
        else:
            print(f"No symbols information to download data. Exit script.")
            sys.exit()

    print("\nDownloading historical data...")
    data = download(market, symbols)
    tools.save_dict_with_timestamp(data, file_prefix, data_dir)

    return data

def save_data(data):
        with open("data.pickle", "wb") as handle:
            pickle.dump(data, handle)

def main():
    args = cli_argparser()
    cache = args.cache
    market = args.market
    if market == "india":
        index, symbols = Index.nse_500()
        #symbols = ['EICHERMOT','HEROMOTOCO','NESTLEIND','ONGC', 'COALINDIA','RELIANCE','BPCL','LTIM','MARUTI','HCLTECH',]
    else:
        index, symbols = Index.dow_jones()
        #symbols = ['meta', 'aapl', 'amzn', 'nflx', 'goog']
    data_dir = f"data/historic_data/{market}"
    file_prifix = f"{index}_data"
    data = load_data(cache, symbols, market, file_prifix, data_dir)

    # Start processing data
    #volatile(args, data)
    screener_one(data, 3)

if __name__ == "__main__":
    main()