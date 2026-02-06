import pandas as pd
import yfinance as yf
from ta.volatility import DonchianChannel
import numpy as np


def generate_donchain_signals(data, screening_period=5, low=20, high=20):
    # calculate donchian channels using ta library
    dc = DonchianChannel(high=data['High'], low=data['Low'], close=data['Close'], window=max(low, high))
    data['don_low'] = dc.donchian_channel_lband()
    data['don_mid'] = dc.donchian_channel_mband()
    data['don_high'] = dc.donchian_channel_hband()
    
    # implement the trading strategy
    #data['long'] = ((data['Close']==data['low'])|(data['Low']==data['low'])).astype('int')
    #data['short'] = ((data['Close']==data['high'])|(data['High']==data['high'])).astype('int')

    data['position'] = np.where((data['Close']==data['don_low'])|(data['Low']==data['don_low']), 1, np.where((data['Close']==data['don_high'])|(data['High']==data['don_high']), -1, 0))
     
    # Get signals for the last 5 days
    signals = data['position'].tail(screening_period)

    return signals

def donchain_screener(data, duration=5):
    '''
    Generate a MACD screener for a given set of stocks based on various signal indicators.
    
    Args:
        data (dict): A dictionary containing "tickers" and "price_data".
        duration (int): The duration in days for which the MACD signals are generated.
        
    Returns:
        dict: A dictionary containing the matching trades for each stock based on the specified conditions in the last `duration` days.
    '''

    tickers = data["tickers"]
    price_data = data["price_data"]
    screener_data = {"BUY": [], "SELL": []}

    for ticker in tickers:
        trade_signals = generate_donchain_signals(price_data[ticker], duration)
        if trade_signals.iloc[-1] == 1:
            screener_data["BUY"].append(ticker)
        elif trade_signals.iloc[-1] == -1:
            screener_data["SELL"].append(ticker)
    return screener_data