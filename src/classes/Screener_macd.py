import numpy as np


def generate_macd_signals(data, macd_screening_period=5):
    # Generate buy/sell signals
    data['position'] = np.where(data['MACD'] > data['MACD_signal'], 1, np.where(data['MACD'] < data['MACD_signal'], -1, 0))

    # Get signals for the last 5 days
    macd_signals = data['position'].tail(macd_screening_period)

    return macd_signals

def macd_screener(data, duration=5):
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
        trade_signals = generate_macd_signals(price_data[ticker], duration)
        if trade_signals.iloc[-1] == 1:
            screener_data["BUY"].append(ticker)
        elif trade_signals.iloc[-1] == -1:
            screener_data["SELL"].append(ticker)
    return screener_data