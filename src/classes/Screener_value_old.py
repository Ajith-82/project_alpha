#!/usr/bin/env python
from classes.Plot_stocks import plot_strategy_multiple
from classes.Screener_ma import add_signal_indicators

PE_THRESHOLD = 15
PB_THRESHOLD = 1.5


def filter_value_stocks(
    ticker, ticker_info, pe_threshold=PE_THRESHOLD, pb_threshold=PB_THRESHOLD
):
    if len(ticker_info) != 0:
        # Get pe_ratio and pe_ratio
        try:
            if ticker_info.get("trailingPE"):
                pe_ratio = float(ticker_info["trailingPE"])
            elif ticker_info.get("forwardPE"):
                pe_ratio = float(ticker_info["forwardPE"])
            else:
                print(f"{ticker} - Can't get values for PeRatio and ForwardPeRatio")
                return False

            if ticker_info.get("priceToBook"):
                pb_ratio = float(ticker_info["priceToBook"])
            else:
                print(f"{ticker} - Can't get values for PbRatio")
                return False
        except ValueError:
            # Handle the case where conversion to float fails
            print("Issues with collecting P/E and P/B ratios values")
            return False

    return pe_ratio < pe_threshold and pb_ratio < pb_threshold


def screener_value(data, out_dir):
    tickers = data["tickers"]
    price_data = data["price_data"]
    valuation_data = data["company_info"]

    plot_tickers = []
    plot_data = {}

    for ticker in tickers:
        try:
            ticker_valuation = valuation_data[ticker]
            price_df = price_data[ticker]
            if filter_value_stocks(ticker, ticker_valuation):
                # print(f"{ticker} is a value stock! (P/E: {PE_THRESHOLD}, P/B: {PB_THRESHOLD})")
                plot_tickers.append(ticker)
                price_df = add_signal_indicators(price_df)
                plot_data[ticker] = price_df
        except Exception as e:
            print(f"Error for {ticker} in processing value_stocks function: {e}")
            pass
    plot_strategy_multiple(plot_tickers, plot_data, out_dir)
    return plot_tickers
