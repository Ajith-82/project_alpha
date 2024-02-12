import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np

def get_candle_type(dailyData):
    return bool(dailyData['Close'].iloc[0] >= dailyData['Open'].iloc[0])

# Find accurate breakout value
def generate_breakout_signals(data, days_to_lookback):
    """
    Generate breakout signals based on the input data and the specified number of days to look back.
    Parameters:
    - data: the input data for generating breakout signals
    - days_to_lookback: the number of days to look back for breakout signals
    Returns:
    - True if a breakout signal is detected, False otherwise
    """
    data = data.fillna(0).replace([np.inf, -np.inf], 0)
    recent = data.tail(1)
    high = round(data['High'].max(), 2)
    close = round(data['Close'].max(), 2)
    recent_close = round(recent['Close'].iloc[0], 2)
    
    if np.isnan(close) or np.isnan(high):
        #save_dict['Breaking-Out'] = 'BO: Unknown'
        return False
    
    if high > close:
        if (high - close) <= (high * 2 / 100):
            #save_dict['Breaking-Out'] = str(close)
            if recent_close >= close:
                return True and get_candle_type(recent)
            return False
        
        no_of_higher_shadows = len(data[data['High'] > close])
        if days_to_lookback / no_of_higher_shadows <= 3:
            #save_dict['Breaking-Out'] = str(high)
            if recent_close >= high:
                return True and get_candle_type(recent)
            return False
        
        #save_dict['Breaking-Out'] = str(close) + ", " + str(high)
        if recent_close >= close:
            return True and get_candle_type(recent)
        return False
    else:
        #save_dict['Breaking-Out'] = str(close)
        if recent_close >= close:
            return True and get_candle_type(recent)
        return False


def breakout_screener(data, duration=5):
    '''
    Generate a breakout screener for a given set of stocks based on various signal indicators.
    
    Args:
        data (dict): A dictionary containing "tickers" and "price_data".
        duration (int): The duration in days for which the breakout signals are generated.
        
    Returns:
        dict: A dictionary containing the matching trades for each stock based on the specified conditions in the last `duration` days.
    '''

    tickers = data["tickers"]
    price_data = data["price_data"]
    screener_data = {"BUY": [], "SELL": []}

    for ticker in tickers:
        breakout_signals = generate_breakout_signals(price_data[ticker], duration)
        if breakout_signals:
            screener_data["BUY"].append(ticker)
        else:
            screener_data["SELL"].append(ticker)
    return screener_data