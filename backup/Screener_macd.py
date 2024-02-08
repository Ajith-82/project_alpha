import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas_ta as ta
import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
import backtrader.strategies as btstrats
from IPython.display import display

class TradeLogger(bt.analyzers.Analyzer):
    """
    Analyzer returning closed trades information.
    """

    def start(self):
        super(TradeLogger, self).start()

    def create_analysis(self):
        self.rets = []
        self.vals = dict()

    def notify_trade(self, trade):
        """Receives trade notifications before each next cycle"""
        if trade.isclosed:
            self.vals = {'Date': self.strategy.datetime.datetime(),
                         'Gross PnL': round(trade.pnl, 2),
                         'Net PnL': round(trade.pnlcomm, 2),
                         'Trade commission': trade.commission,
                         'Trade duration (in days)': (trade.dtclose - trade.dtopen)
            }
            self.rets.append(self.vals)

    def get_analysis(self):
        return self.rets


def set_and_run(ticker, data, strategy, initial_cash, commission, stake):
    '''
    Run backtest and return buy/sell signals for the last 5 days
    '''
    # Initialize backtrader engine, add the strategy and initial capital
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)
    cerebro.adddata(data)
    cerebro.broker.setcash(initial_cash)

    # Set the broker commission
    cerebro.broker.setcommission(commission) 

    # Number of shares to buy/sell
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)

    # Add evaluation metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, timeframe=bt.TimeFrame.Days, compression=1, factor=365, annualize=True)
    cerebro.addanalyzer(TradeLogger, _name="trade_logger")

    try:
        results = cerebro.run()
        print(dir(results[0]))
        print(dir(results[0].getpositions)

        # Plot the backtest results
        plt.rcParams['figure.figsize'] = (16, 8)
        fig = cerebro.plot(barupfill=False,
                           bardownfill=False,
                           style='candle',
                           plotdist=0.5, 
                           volume=True,
                           barup='green',
                           valuetags=False,
                           subtxtsize=8)
        
        # Save the plot as an SVG file
        plt.savefig(f'data/processed_data/screener_macd/{ticker}.svg', dpi=300)  

    except Exception as e:
        # Handle any exceptions and return the error message
        return str(e)

class MACDStrategy(bt.Strategy):
    params = (('short_period', 25),
              ('long_period', 51),
              ('signal_period', 9)
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=self.params.short_period,
                                       period_me2=self.params.long_period,
                                       period_signal=self.params.signal_period)
        
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal, plot=True, subplot=True)

    def next(self):
        if not self.position:
            if self.macd_cross > 0:
                self.buy()
        else:
            if self.macd_cross < 0:
                self.sell()

def screener_macd(data):
    '''
    Generate a MACD screener for a given set of stocks based on various signal indicators.
    
    Args:
        data (dict): A dictionary containing "tickers" and "price_data".
        
    Returns:
        dict: A dictionary containing the matching trades for each stock based on the specified conditions in last 5 days.
    '''

    tickers = data["tickers"]
    price_data = data["price_data"]
    screener_data = {}

    for ticker in tickers:
        # Add the MACD indicator
        price_data[ticker].index = pd.to_datetime(price_data[ticker].index)
        data_bt = bt.feeds.PandasData(dataname=price_data[ticker])
        startcash = 1000
        commission = 0.001
        stake = 1
        # Run the strategy & plot the results
        trade_signals =set_and_run(ticker,
                    data_bt,
                    MACDStrategy,
                    startcash,
                    commission,
                    stake)
        print(ticker)
        print(trade_signals)
        screener_data[ticker] = trade_signals
        

    return screener_data