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


def plot_strategy_one(ticker, df, max_bars=200):
    """
    Plot RSI, Price and SMAs, and MACD for a given stock.

    Parameters:
    - ticker: str, the stock ticker symbol
    - df: pd.DataFrame, stock data
    - max_bars: int, maximum number of bars to display on the x-axis
    """

    # Calculate RSI
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.tail(200)

    # Plotting
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3, 1])

    # Add subplots using the gridspec layout
    axes = [plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])]

    # Plotting RSI
    axes[0].plot(df['RSI'], label='RSI', color='purple')
    axes[0].axhline(70, linestyle='--', color='red', alpha=0.5)  # Overbought threshold
    axes[0].axhline(30, linestyle='--', color='green', alpha=0.5)  # Oversold threshold
    axes[0].axhline(50, linestyle='--', color='black', alpha=0.5)
    axes[0].legend()

    # Plotting Price and SMAs
    axes[1].plot(df['Close'], label='Close Price', color='black')
    axes[1].plot(df['SMA_10'], label='SMA 10', color='green')
    axes[1].plot(df['SMA_30'], label='SMA 30', color='yellow')
    axes[1].plot(df['SMA_50'], label='SMA 50', color='orange')
    axes[1].plot(df['SMA_200'], label='SMA 200', color='blue')
    axes[1].legend()

    # Plotting MACD
    axes[2].bar(df.index, df['MACD_hist'], label='MACD Histogram', color='gray')
    axes[2].plot(df['MACD'], label='MACD', color='blue')
    axes[2].plot(df['MACD_signal'], label='Signal Line', color='orange')
    axes[2].axhline(0, linestyle='--', color='gray', alpha=0.5)
    axes[2].legend()

    # Disable x-axis labels for subplots [0] and [1]
    axes[0].set_xticks([])
    axes[1].set_xticks([])

    # Set individual titles for each subplot
    axes[0].set_title(f'{ticker} Report')

    # Customize x-axis date labels
    axes[2].xaxis.set_major_locator(plt.MaxNLocator(max_bars // 10))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Adjust space between subplots
    plt.subplots_adjust(hspace=0)

    plt.xlabel('Date')
    plt.tight_layout(pad=0)

    # Save the figure
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{ticker}.png', dpi=fig.dpi)