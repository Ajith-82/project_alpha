import matplotlib.pyplot as plt
import yfinance as yf

# Define the stock symbol and date range
symbol = "TSLA"
start_date = "2023-01-01"
end_date = "2023-12-31"

# Fetch the data
tesla_data = yf.download(symbol, start=start_date, end=end_date)

# Calculate daily returns
tesla_data['Daily_Return'] = tesla_data['Adj Close'].pct_change()

# Calculate volatility (standard deviation of daily returns)
volatility = tesla_data['Daily_Return'].std()

# Calculate average daily trading volume
average_volume = tesla_data['Volume'].mean()


# Define trading thresholds
volatility_threshold = 0.03# Adjust as needed
volume_threshold = 5000000  # Adjust as needed

# Generate trading signals
tesla_data['Buy_Signal'] = (tesla_data['Daily_Return'].rolling(window=14).std(
) > volatility_threshold) & (tesla_data['Volume'] > volume_threshold)
tesla_data['Sell_Signal'] = (tesla_data['Daily_Return'].rolling(window=14).std(
) < volatility_threshold) | (tesla_data['Volume'] < volume_threshold)


# Plotting
plt.figure(figsize=(12, 6))
plt.plot(tesla_data.index,
         tesla_data['Adj Close'], label='Tesla Stock Price', alpha=0.7)
plt.scatter(tesla_data.index[tesla_data['Buy_Signal']], tesla_data['Adj Close']
            [tesla_data['Buy_Signal']], marker='^', color='g', label='Buy Signal')
plt.scatter(tesla_data.index[tesla_data['Sell_Signal']], tesla_data['Adj Close']
            [tesla_data['Sell_Signal']], marker='v', color='r', label='Sell Signal')
plt.title('Tesla Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()