#!/usr/bin/env python
import datetime as dt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
import os.path
import pickle

from classes.Download import download
from classes.Volatile import volatile
from classes.Screener_one import screener_one
import classes.IndexListFetcher as Index


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

def load_data(cache, symbols: list, market: str):
    if cache and os.path.exists("data.pickle"):
        print("\nLoading historical data...")
        with open("data.pickle", "rb") as handle:
            data = pickle.load(handle)
        print("Data has been saved to {}/{}.".format(os.getcwd(), "data.pickle"))
    else:
        if symbols is None:
            with open("symbols_list.txt", "r") as my_file:
                symbols = my_file.readlines()[0].split(" ")
        print("\nDownloading historical data...")
        data = download(market, symbols)
    return data

def save_data(data):
        with open("data.pickle", "wb") as handle:
            pickle.dump(data, handle)

def main():
    args = cli_argparser()
    cache = args.cache
    market = args.market
    if market == "india":
        symbols = Index.nse_500()
        #symbols = ['EICHERMOT','HEROMOTOCO','NESTLEIND','ONGC', 'COALINDIA','RELIANCE','BPCL','LTIM','MARUTI','HCLTECH',]
    else:
        symbols = Index.sp_500()
        symbols = ['meta', 'aapl', 'amzn', 'nflx', 'goog']
    data = load_data(cache, symbols, market)
    if not cache:
        save_data(data)
    #volatile(args, data)
    screener_one(data, 50)

if __name__ == "__main__":
    main()