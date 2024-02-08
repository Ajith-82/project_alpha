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

def generate_trade_signals():
    class DonchianChannels(bt.Indicator):
        '''
        Params Note:
        - `lookback` (default: -1)
            If `-1`, the bars to consider will start 1 bar in the past and the
            current high/low may break through the channel.
            If `0`, the current prices will be considered for the Donchian
            Channel. This means that the price will **NEVER** break through the
            upper/lower channel bands.
        '''

        alias = ('DCH', 'DonchianChannel',)

        lines = ('dcm', 'dch', 'dcl',)  # dc middle, dc high, dc low
        params = dict(
            period=20,
            lookback=-1,  # consider current bar or not
        )

        plotinfo = dict(subplot=False)  # plot along with data
        plotlines = dict(
            dcm=dict(ls='--'),  # dashed line
            dch=dict(_samecolor=True),  # use same color as prev line (dcm)
            dcl=dict(_samecolor=True),  # use same color as prev line (dch)
        )

        def __init__(self):
            hi, lo = self.data.high, self.data.low
            if self.p.lookback:  # move backwards as needed
                hi, lo = hi(self.p.lookback), lo(self.p.lookback)

            self.l.dch = bt.ind.Highest(hi, period=self.p.period)
            self.l.dcl = bt.ind.Lowest(lo, period=self.p.period)
            self.l.dcm = (self.l.dch + self.l.dcl) / 2.0  # avg of the above

    class DonChainStrategy(bt.Strategy):
        params = (('lower_length', 20),
                ('upper_length', 20)
                )

        def __init__(self):
            self.donchain = DonchianChannels(self.data, lower_length=self.params.lower_length,
                                                    upper_length=self.params.upper_length)

        def next(self):
            if not self.position:
            if self.data.close == self.donchain.low | self.data.low == self.donchain.low:
                self.buy()
            else:
            if self.data.close == self.donchain.high | self.data.high == self.donchain.high:
                self.sell()