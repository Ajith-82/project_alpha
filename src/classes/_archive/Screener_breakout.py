import numpy as np

def breakouts_screener(stock_data, min_ave_volume):
    '''
    A function that accepts stocks price and returns True if at least one breakout candle in the past 5 days

    Args:
        stock_data (pd.DataFrame): A dataframe containing stock data
        min_ave_volume (int): The minimum average volume

    Returns:
        bool: True if at least one breakout candle in the past 5 days, False otherwise        
    '''

    try:
        latest_25_candles = stock_data.tail(25).copy()
        latest_25_candles = latest_25_candles.drop(['Adj Close', 'Dividends', 'Stock Splits', 'SMA_10', 'SMA_30', 'SMA_50', 'SMA_200', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI'], axis=1)

        if latest_25_candles.empty:
            return False

        latest_25_candles['SellingPressure'] = latest_25_candles['High'] - latest_25_candles['Close']
        latest_25_candles['O_to_C'] = latest_25_candles['Close'] - latest_25_candles['Open']
        latest_25_candles['OC_20D_Mean'] = latest_25_candles['O_to_C'].rolling(20).mean()
        latest_25_candles['OC_perc_from_20D_Mean'] = 100*(latest_25_candles['O_to_C'] - latest_25_candles['OC_20D_Mean'])/latest_25_candles['OC_20D_Mean']
        latest_25_candles['MaxOC_Prev10'] = latest_25_candles['O_to_C'].rolling(10).max()
        latest_25_candles['Volume_20D_Mean'] = latest_25_candles['Volume'].rolling(20).mean()
        latest_25_candles['Volume_perc_from_20D_Mean'] = 100*(latest_25_candles['Volume'] - latest_25_candles['Volume_20D_Mean'])/latest_25_candles['Volume_20D_Mean']

        latest_5_candles = latest_25_candles.tail(5)
        condition = (latest_5_candles['O_to_C'] >= 0.0) & (latest_5_candles['O_to_C'] == latest_5_candles['MaxOC_Prev10']) & (latest_5_candles['SellingPressure']/latest_5_candles['O_to_C'] <= 0.40) & (latest_5_candles['OC_perc_from_20D_Mean'] >= 100.0) & (latest_5_candles['Volume_perc_from_20D_Mean'] >= 50.0)
        breakouts = latest_5_candles[condition].reset_index(drop=True)

        ave_volume = latest_5_candles['Volume'].mean()

        if ave_volume >= min_ave_volume and not breakouts.empty:
            return True
        else:
            return False
        
    except Exception as e:
        return f"An error occurred while screening for breakout stocks: {e}"


def breakout_screener(data, tickers):
    '''
    Generate a breakout screener for a given set of stocks based on various signal indicators.
    
    Args:
        data (dict): A dictionary containing "tickers" and "price_data".
        duration (int): The duration in days for which the breakout signals are generated.
        
    Returns:
        dict: A dictionary containing the matching trades for each stock based on the specified conditions in the last `duration` days.
    '''

    #tickers = data["tickers"]
    price_data = data["price_data"]
    screener_data = {"BUY": [], "SELL": []}

    for ticker in tickers:
        breakout_signals = breakouts_screener(price_data[ticker], min_ave_volume=100000)
        if breakout_signals:
            screener_data["BUY"].append(ticker)
        else:
            screener_data["SELL"].append(ticker)
    return screener_data