import numpy as np
import os
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Create a wrapper module to provide a consistent interface
class talib:
    """Wrapper class to provide pandas_ta-like interface using ta library."""
    
    @staticmethod
    def ema(close, length):
        return EMAIndicator(close=close, window=length).ema_indicator()
    
    @staticmethod 
    def sma(close, length):
        return SMAIndicator(close=close, window=length).sma_indicator()
    
    @staticmethod
    def ma(close, length):
        return SMAIndicator(close=close, window=length).sma_indicator()
    
    @staticmethod
    def macd(close, fast, slow, signal):
        macd_ind = MACD(close=close, window_fast=fast, window_slow=slow, window_sign=signal)
        import pandas as pd
        return pd.DataFrame({
            f'MACD_{fast}_{slow}_{signal}': macd_ind.macd(),
            f'MACDs_{fast}_{slow}_{signal}': macd_ind.macd_signal(),
            f'MACDh_{fast}_{slow}_{signal}': macd_ind.macd_diff()
        })
    
    @staticmethod
    def rsi(close, length):
        return RSIIndicator(close=close, window=length).rsi()
    
    @staticmethod
    def cci(high, low, close, length):
        from ta.trend import CCIIndicator
        return CCIIndicator(high=high, low=low, close=close, window=length).cci()
    
    # Candlestick pattern methods - return False as placeholder since ta library doesn't support them
    @staticmethod
    def cdl_pattern(open, high, low, close, pattern):
        return None


class ScreenerTA:

    @staticmethod
    def EMA(close, timeperiod):
        try:
            return talib.ema(close,timeperiod)
        except Exception as e:
            return talib.EMA(close,timeperiod)

    @staticmethod
    def SMA(close, timeperiod):
        try:
            return talib.sma(close,timeperiod)
        except Exception as e:
            return talib.SMA(close,timeperiod)
        
    @staticmethod
    def MA(close, timeperiod):
        try:
            return talib.ma(close,timeperiod)
        except Exception as e:
            return talib.MA(close,timeperiod)

    @staticmethod
    def MACD(close, fast, slow, signal):
        try:
            return talib.macd(close,fast,slow,signal)
        except Exception as e:
            return talib.MACD(close,fast,slow,signal)

    @staticmethod
    def RSI(close, timeperiod):
        try:
            return talib.rsi(close,timeperiod)
        except Exception as e:
            return talib.RSI(close,timeperiod)
    
    @staticmethod
    def CCI(high, low, close, timeperiod):
        try:
            return talib.cci(high, low, close,timeperiod)
        except Exception as e:
            return talib.CCI(high, low, close,timeperiod)
    
   
    @staticmethod
    def CDLMORNINGSTAR(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'morningstar').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLMORNINGSTAR(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False

    @staticmethod
    def CDLMORNINGDOJISTAR(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'morningdojistar').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLMORNINGDOJISTAR(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLEVENINGSTAR(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'eveningstar').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLEVENINGSTAR(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLEVENINGDOJISTAR(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'eveningdojistar').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLEVENINGDOJISTAR(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLLADDERBOTTOM(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'ladderbottom').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLLADDERBOTTOM(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDL3LINESTRIKE(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'3linestrike').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDL3LINESTRIKE(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDL3BLACKCROWS(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'3blackcrows').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDL3BLACKCROWS(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDL3INSIDE(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'3inside').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDL3INSIDE(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDL3OUTSIDE(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'3outside').tail(1).values[0][0]
            except Exception as e:
                return talib.CDL3OUTSIDE(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDL3WHITESOLDIERS(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'3whitesoldiers').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDL3WHITESOLDIERS(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLHARAMI(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'harami').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLHARAMI(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLHARAMICROSS(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'haramicross').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLHARAMICROSS(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLMARUBOZU(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'marubozu').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLMARUBOZU(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLHANGINGMAN(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'hangingman').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLHANGINGMAN(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLHAMMER(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'hammer').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLHAMMER(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLINVERTEDHAMMER(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'invertedhammer').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLINVERTEDHAMMER(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLSHOOTINGSTAR(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'shootingstar').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLSHOOTINGSTAR(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLDRAGONFLYDOJI(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'dragonflydoji').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLDRAGONFLYDOJI(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLGRAVESTONEDOJI(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'gravestonedoji').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLGRAVESTONEDOJI(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
    
    @staticmethod
    def CDLDOJI(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'doji').tail(1).values[0][0] != 0
            except Exception as e:
                return talib.CDLDOJI(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
        
    
    @staticmethod
    def CDLENGULFING(open, high, low, close):
        try:
            try:
                return talib.cdl_pattern(open,high,low,close,'engulfing').tail(1).values[0][0]
            except Exception as e:
                return talib.CDLENGULFING(open,high,low,close).tail(1).item() != 0
        except AttributeError:
            return False
        