import requests
import pandas as pd
from io import StringIO
from src.classes.IndexListFetcher import sp_500

PE_THRESHOLD = 15
PB_THRESHOLD = 1.5

class YahooFinanceError(Exception):
    pass

def get_stock_data(ticker):
    base_url = 'https://finance.yahoo.com'
    url = f"{base_url}/quote/{ticker}/key-statistics?p={ticker}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url=url, headers=headers, timeout=5)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

        if "Will be right back" in response.text:
            raise YahooFinanceError("*** YAHOO! FINANCE is currently down! ***\n")

        dfs = pd.read_html(StringIO(response.text))

        if not dfs:
            raise YahooFinanceError(f"No data found for {ticker}")

        df = dfs[0].rename(columns={0: 'Metric', 1: 'Value'})
        return df

    except requests.exceptions.RequestException as e:
        raise YahooFinanceError(f"Error fetching data for {ticker}: {e}")

def filter_value_stocks(df, pe_threshold=PE_THRESHOLD, pb_threshold=PB_THRESHOLD):
    # Filter rows based on the metric
    pe_row = df[df['Metric'] == 'Trailing P/E']
    pb_row = df[df['Metric'] == 'Price/Book (mrq)']

    # Check if the metric is present and get the corresponding values
    if not pe_row.empty and not pb_row.empty:
        pe_ratio_str = pe_row['Value'].iloc[0]
        pb_ratio_str = pb_row['Value'].iloc[0]

        # Convert the string values to numeric
        try:
            pe_ratio = float(pe_ratio_str)
            pb_ratio = float(pb_ratio_str)
        except ValueError:
            # Handle the case where conversion to float fails
            return False

        return pe_ratio < pe_threshold and pb_ratio < pb_threshold

    # Return False if the metric is not present
    return False

#tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'META']
index, tickers = sp_500()
for ticker in tickers:
    try:
        df = get_stock_data(ticker)
        if filter_value_stocks(df):
            print(f"{ticker} is a value stock! (P/E: {PE_THRESHOLD}, P/B: {PB_THRESHOLD})")
        #else:
            #print(f"{ticker} is not a value stock. (P/E: {PE_THRESHOLD}, P/B: {PB_THRESHOLD})")
    except YahooFinanceError as e:
        print(e)
