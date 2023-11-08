import pandas as pd
from yahooquery import Ticker
from datetime import datetime as dt
from datetime import timedelta as delta

def convert_yahooquery_to_yfinance(df_yahooquery):
    # Step 1: Reset the index to make 'Date' a regular column
    df_yahooquery.reset_index(inplace=True)

    # Step 2: Rename columns
    df_yahooquery.rename(columns={
        'date': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'adjclose': 'Adj Close',
        'volume': 'Volume'
    }, inplace=True)

    # Step 3: Convert "Date" to a string in "YYYY-MM-DD" format
    df_yahooquery['Date'] = pd.to_datetime(df_yahooquery['Date']).dt.strftime('%Y-%m-%d')

    # Step 4: Set "Date" as the index
    df_yahooquery.set_index('Date', inplace=True)

    # Step 5: Remove the "Ticker" column
    df_yahooquery.drop('symbol', axis=1, inplace=True)

    return df_yahooquery

# Example usage:
# Assuming you have a DataFrame df_yahooquery from yahooquery.history
from yahooquery import Ticker
stock = Ticker("SBIN.NS")
_df = stock.history(interval='1d', 
                    start='2022-01-01', 
                    end=(dt.now() + delta(1)).strftime('%Y-%m-%d'))
print(_df)
df_yfinance = convert_yahooquery_to_yfinance(_df)
print(df_yfinance)