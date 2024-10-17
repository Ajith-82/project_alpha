#!/usr/bin/env python
from curses import raw
import os

from classes.Send_email import send_email_volatile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
import os.path

from classes.Download import load_data, load_volatile_data
from classes.Volatile import volatile
from classes.Screener_ma import screener_ma
from classes.Screener_macd import macd_screener
from classes.Screener_donchain import donchain_screener
#from classes.Screener_value import screener_value
from classes.Screener_breakout import breakout_screener
import classes.IndexListFetcher as Index
import classes.Tools as tools
from classes.Plot_stocks import create_plot_and_email_batched
from classes.Screener_trendline import trendline_screener
from classes. Send_email import send_email_volatile

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
    """
    Screener value charts for a given market, index, and list of symbols.

    Args:
        cache: The cache object for data storage.
        market (str): The market for which the data is being processed.
        index (str): The index for which the data is being processed.
        symbols (list): The list of symbols for which the data is being processed.

    Returns:
        None
    """
    historic_data_dir = f"data/historic_data/{market}"
    if not os.path.exists(historic_data_dir):
        os.mkdir(historic_data_dir)
    
    file_prefix = f"{index}_data"
    data = load_data(cache, symbols, market, file_prefix, historic_data_dir)
    
    price_data = data["price_data"]
    value_symbols = data["tickers"]
    
    processed_data_dir = f"data/processed_data/{index}"
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    
    create_plot_and_email_batched("IND_screener_value_stocks", market, value_symbols, price_data, processed_data_dir)


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
    print("\nStarting Volatility based screening...")
    volatile_data = load_volatile_data(market, data)
    volatile_df = volatile(args, volatile_data)
    volatile_symbols_top = volatile_df["SYMBOL"].head(200).tolist()
    volatile_symbols_bottom = volatile_df["SYMBOL"].tail(200).tolist()
    #send_email_volatile(market, "data/processed_data/volatile")
    print("\nFinished Volatility based screening...")

    '''
    # Start MA screener
    ma_screener_out_dir = "data/processed_data/screener_ma"
    print("\nStarting MA based screening...")
    ma_screener_out = screener_ma(data, screener_dur)
    ma_screener_symbols =list(ma_screener_out.keys())
    create_plot_and_email_batched("MA screener", market, ma_screener_symbols, data, ma_screener_out_dir)
    print("\nFinished MA based screening...")

    # Start MACD screener
    macd_screener_out_dir = "data/processed_data/screener_macd"
    print("\nStarting MACD based screening...")
    macd_screener_out = macd_screener(data, screener_dur)
    macd_screener_symbols = macd_screener_out["BUY"]
    create_plot_and_email_batched("MACD screener", market, macd_screener_symbols, data, macd_screener_out_dir)
    print("\nFinished MACD based screening...")

    # Start Donchain screener
    macd_screener_out_dir = "data/processed_data/screener_donchain"
    print("\nStarting Donchain based screening...")
    donchain_screener_out = donchain_screener(data, screener_dur)
    donchain_screener_symbols = donchain_screener_out["BUY"]
    create_plot_and_email_batched("Donchain screener", market, donchain_screener_symbols, data, macd_screener_out_dir)
    print("\nFinished Donchain based screening...")
    
    '''
    # Start breakout screener
    breakout_screener_out_dir = "data/processed_data/screener_breakout"
    print("\nStarting Breakout based screening...")
    breakout_screener_out = breakout_screener(data, volatile_symbols_bottom)
    breakout_screener_out_symbols = breakout_screener_out["BUY"]
    create_plot_and_email_batched("Breakout screener", market, breakout_screener_out_symbols, data, breakout_screener_out_dir)
    tools.save_screener_results_to_csv(market, "screener_breakout", breakout_screener_out_symbols)
    print("\nFinished Breakout based screening...")
    
    # Start trend screener
    trend_screener_out_dir = "data/processed_data/screener_trend"
    trend_screener_history = "data/processed_data/screener_trend_history"
    print("\nStarting trend based screening...")
    trend_screener_out = trendline_screener(data, volatile_symbols_top, screener_dur)
    trend_screener_out_symbols = [ticker for ticker, _ in trend_screener_out['Trend']]
    create_plot_and_email_batched("Trend screener", market, trend_screener_out_symbols, data, trend_screener_out_dir)
    trend_history_file = tools.save_screener_results_to_csv(market, "screener_trend", trend_screener_out_symbols)
    trend_common = tools.find_common_symbols(trend_history_file, 5)
    if len(trend_common) > 0:
        create_plot_and_email_batched("Trending stocks in last 5 days", market, trend_common, data, trend_screener_history)
    print("\nFinished trend based screening...")

if __name__ == "__main__":
    main()
