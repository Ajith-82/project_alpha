#!/usr/bin/env python
from curses import raw
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
import os.path
import pickle

from classes.Download import load_data, load_volatile_data
from classes.Volatile import volatile
from classes.Screener_ma import screener_ma
from classes.Screener_macd import macd_screener
from classes.Screener_donchain import donchain_screener
from classes.Screener_value import screener_value
import classes.IndexListFetcher as Index
import classes.Tools as tools
from classes.Plot_stocks import create_plot_and_email


def cli_argparser():
    cli = ArgumentParser(
        "Volatile: your day-to-day trading companion.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    cli.add_argument("-s", "--symbols", type=str, nargs="+", help=SUPPRESS)
    cli.add_argument(
        "--rank",
        type=str,
        default="growth",
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
        "--save-table",
        action="store_true",
        default=True,
        help="Save prediction table in csv format.",
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
        default=True,
        help="Use cached data and parameters if available.",
    )
    cli.add_argument(
        "--value",
        action="store_true",
        help="Plot and send value stocks from external source.",
    )
    return cli.parse_args()



def screener_value_charts(cache, market: str, index: str, symbols: list):
    if not os.path.exists(f"data/historic_data/{market}"):
        os.mkdir(f"data/historic_data/{market}")
    data_dir = f"data/historic_data/{market}"
    file_prifix = f"{index}_data"
    data = load_data(cache, symbols, market, file_prifix, data_dir)
    price_data = data["price_data"]
    value_symbols = data["tickers"]
    plot_data = {}
    for symbol in value_symbols:
        plot_data[symbol] = add_signal_indicators(price_data[symbol])

    if not os.path.exists(f"data/processed_data/{index}"):
        os.mkdir(f"data/processed_data/{index}")
    screener_out_dir = f"data/processed_data/{index}"
    plot_strategy_multiple(value_symbols, plot_data, screener_out_dir)
    send_email(market, "External screener charts", screener_out_dir)

def add_indicator_plot(screener: str, market: str, symbols: list, data: dict, out_dir: str):
    raw_data = data["price_data"]
    plot_data = {}
    for symbol in symbols:
        plot_data[symbol] = add_signal_indicators(raw_data[symbol])

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    plot_strategy_multiple(symbols, plot_data, out_dir)
    send_email(market, screener, out_dir)


def main():
    # Cleanup report directories
    tools.cleanup_directory_files("data/processed_data")

    args = cli_argparser()
    cache = args.cache
    market = args.market

    # Load data
    if market == "india":
        index, symbols = Index.nse_500()
        screener_dur = 3
        if args.value:
            value_index, value_symbols = Index.ind_screener_value_stocks()
            screener_value_charts(cache, market, value_index, value_symbols)
    else:
        index, symbols = Index.sp_500()
        screener_dur = 3

    if not os.path.exists(f"data/historic_data/{market}"):
        os.mkdir(f"data/historic_data/{market}")
    data_dir = f"data/historic_data/{market}"
    file_prifix = f"{index}_data"
    data = load_data(cache, symbols, market, file_prifix, data_dir)

    # Start processing data
    '''
    screener_volatile_top = "data/processed_data/volatile_top"
    screener_volatile_bottom = "data/processed_data/volatile_bottom"
    print("\nStarting Volatility based screening...")
    volatile_data = load_volatile_data(market, data)
    volatile_df = volatile(args, volatile_data)
    volatile_symbols_top = volatile_df["SYMBOL"].head(100).tolist()
    volatile_symbols_bottom = volatile_df["SYMBOL"].tail(100).tolist()
    create_plot_and_email("Volatile top 100", market, volatile_symbols_top, data, screener_volatile_top)
    create_plot_and_email("Volatile bottom 100", market, volatile_symbols_bottom, data, screener_volatile_bottom)
    print("\nFinished Volatility based screening...")

    # Start MA screener
    ma_screener_out_dir = "data/processed_data/screener_ma"
    print("\nStarting MA based screening...")
    ma_screener_out = screener_ma(data, screener_dur)
    ma_screener_symbols =list(ma_screener_out.keys())
    create_plot_and_email("MA screener", market, ma_screener_symbols, data, ma_screener_out_dir)
    print("\nFinished MA based screening...")
    '''

    # Start MACD screener
    macd_screener_out_dir = "data/processed_data/screener_macd"
    print("\nStarting MACD based screening...")
    macd_screener_out = macd_screener(data, screener_dur)
    macd_screener_symbols = macd_screener_out["BUY"]
    create_plot_and_email("MACD screener", market, macd_screener_symbols, data, macd_screener_out_dir)
    print("\nFinished MACD based screening...")

    # Start Donchain screener
    macd_screener_out_dir = "data/processed_data/screener_donchain"
    print("\nStarting Donchain based screening...")
    donchain_screener_out = donchain_screener(data, screener_dur)
    donchain_screener_symbols = donchain_screener_out["BUY"]
    print(donchain_screener_symbols)
    create_plot_and_email("Donchain screener", market, donchain_screener_symbols, data, macd_screener_out_dir)
    print("\nFinished Donchain based screening...")

if __name__ == "__main__":
    main()
