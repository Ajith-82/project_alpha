import numpy as np
from scipy.signal import argrelextrema
from classes.Tools import SuppressOutput

class StockDataNotAdequate(Exception):
    pass

def find_trend(data, lookback_days=5):
    """
    Finds the trend of the given stock data based on the angle of the slope.
    
    Args:
        data: The stock data to analyze.
        lookback_days: The number of days to look back for trend analysis (default is 5).
        
    Returns:
        A string representing the trend, which can be 'Unknown', 'Sideways', 'Weak Up', 'Strong Up', 'Weak Down', or 'Strong Down'.
    """
    # Ensure enough data for analysis
    if len(data) < lookback_days:
        raise StockDataNotAdequate("Not enough data for analysis. Please provide at least " + str(lookback_days) + " days of data.")

    trend_data = data.tail(lookback_days).copy()
    trend_data = trend_data.set_index(np.arange(len(trend_data)))
    trend_data.fillna(0, inplace=True)
    trend_data.replace([np.inf, -np.inf], 0, inplace=True)
    try:
        trend_data['tops'] = trend_data['Close'].iloc[list(argrelextrema(np.array(trend_data['Close']), np.greater_equal, order=1)[0])]
    except ValueError as e:
        print("Error identifying peaks:", e)
        trend_data['tops'] = 0

    trend_data.fillna(0, inplace=True)
    trend_data.replace([np.inf, -np.inf], 0, inplace=True)
    try:
        #slope, c = np.polyfit(trend_data.index[trend_data.tops > 0], trend_data['tops'][trend_data.tops > 0], 1)
        slope, _ = np.polyfit(trend_data.index, trend_data['Close'], 1)
        #peak_slope, _ = np.polyfit(trend_data.index[trend_data.tops > 0], trend_data['tops'][trend_data.tops > 0], 1)

    except Exception as e:
        slope, _, peak_slope = 0, 0, 0
    #angle = np.rad2deg(np.arctan((slope + peak_slope) / 2))
    angle = np.rad2deg(np.arctan(slope))
    if angle == 0:
        return 'Unknown'
    elif -30 <= angle <= 30:
        return 'Sideways'
    elif 30 < angle < 60:
        return 'Weak Up'
    elif angle >= 60:
        return 'Strong Up'
    elif -60 < angle < -30:
        return 'Weak Down'
    elif angle <= -60:
        return 'Strong Down'

def trendline_screener(data, tickers, duration=5):
    '''
    Generate a screener for a given set of stocks based on various signal indicators.
    
    Args:
        data (dict): A dictionary containing "tickers" and "price_data".
        duration (int): The duration in days for which the MACD signals are generated.
        
    Returns:
        dict: A dictionary containing the matching trades for each stock based on the specified conditions in the last `duration` days.
    '''

    #tickers = data["tickers"]
    price_data = data["price_data"]
    screener_data = {"Trend": []}

    for ticker in tickers:
        trend = find_trend(price_data[ticker], duration)
        if trend:
            screener_data["Trend"].append((ticker, trend))
    filtered_data = {'Trend': [(ticker, trend) for ticker, trend in screener_data['Trend'] if trend in ('Strong Up')]}
    return filtered_data
