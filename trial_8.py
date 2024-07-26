import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import os

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

# Define selection handler
selected_periods = []

def onselect(xmin, xmax):
    selected_period = data.loc[(data.index >= xmin) & (data.index <= xmax)]
    selected_periods.append(selected_period)
    print(f'Selected period from {xmin} to {xmax}')

# Fetch stock data
ticker = 'AAPL'
start_date = '2022-01-01'
end_date = '2022-12-31'
data = fetch_stock_data(ticker, start_date, end_date)

# Plot data and enable selection
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['Close'], label='Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title(f'{ticker} Close Price')
plt.legend()

# SpanSelector to select the range on the plot
span = SpanSelector(ax, onselect, 'horizontal', useblit=True, span_stays=True, rectprops=dict(alpha=0.5, facecolor='red'))

plt.show()

# Save selected periods
output_dir = 'selected_periods'
os.makedirs(output_dir, exist_ok=True)

for i, period in enumerate(selected_periods):
    period.to_csv(os.path.join(output_dir, f'selected_period_{i+1}.csv'))

print(f'Saved {len(selected_periods)} selected periods.')
