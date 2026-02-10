#!/usr/bin/env python
import os
import math
from token import DOUBLESLASH
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
from classes.Send_email import send_email
import classes.Tools as tools


# Sets padding for figures
def set_padding(fig):
    fig.update_layout(
        margin=go.layout.Margin(r=10, b=10)  # right margin
    )  # bottom margin


# Adds the range selector to given figure
def add_range_selector(fig):
    """
    Update the layout of the figure to include a range selector for the x-axis.
    
    :param fig: The figure object to update.
    :type fig: plotly.graph_objs.Figure
    
    :return: None
    """
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
            type="date"
        ),
        xaxis2_type="date"
    )


# Adds the volume chart to row 2, column 1
def add_volume_chart(fig, df):
    """
    Adds the volume bar chart to the given figure using data from the provided DataFrame.
    :param fig: the plotly figure to which the bar chart will be added
    :param df: the DataFrame containing the data for the bar chart
    """
    # Define color based on the price change
    colors = ["#9C1F0B" if row["Open"] - row["Close"] >= 0 else "#2B8308" for index, row in df.iterrows()]

    # Add volume bar chart to the figure
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], showlegend=False, marker_color=colors), row=2, col=1)


# Adds the MACD chart to row 3, column 1
def add_macd_chart(fig, df):
    """
    Adds MACD chart to the given figure using the provided dataframe.

    Parameters:
    - fig: The figure to add the MACD chart to.
    - df: The dataframe containing the MACD data.

    Returns:
    - None
    """
    # Use list comprehension to calculate colors for the Bar chart
    colors = ["green" if row["MACD_hist"] >= 0 else "red" for _, row in df.iterrows()]

    # Add Bar chart trace
    fig.add_trace(
        go.Bar(x=df.index, y=df["MACD_hist"], showlegend=False, marker_color=colors),
        row=3, col=1,
    )

    # Add MACD trace
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="black", width=2)),
        row=3, col=1,
    )

    # Add MACD signal trace
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD_signal"], name="MACD_signal", line=dict(color="blue", width=1)),
        row=3, col=1,
    )



# Adds the RSI chart to row 4, column 1
def add_rsi_chart(fig, df):
    """
    Add a relative strength index (RSI) chart to the given figure.

    Parameters:
    - fig: the plotly figure to add the RSI chart to
    - df: the dataframe containing the RSI data

    Returns:
    None
    """
    rsi_trace = go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple", width=2))
    fig.add_trace(rsi_trace, row=4, col=1)

    # Add horizontal lines for Y=30 and Y=70
    horizontal_line = dict(type="line", x0=df.index[0], x1=df.index[-1], line=dict(color="green", width=1))
    fig.add_shape(horizontal_line, y0=30, y1=30, row=4, col=1)
    fig.add_shape(horizontal_line, y0=70, y1=70, row=4, col=1)

    fig.update_yaxes(range=[0, 100], row=4, col=1)

# Adds the RSI chart to row 1, column 1
def add_donchain_chart(fig, df, low=20, high=20):
    """
    Add a Donchain chart to the given figure.

    Parameters:
    - fig: the plotly figure to add the RSI chart to
    - df: the dataframe containing the RSI data

    Returns:
    None
    """
    # calculate donchian channels using ta library
    try:
        from ta.volatility import DonchianChannel
        donchian = DonchianChannel(high=df["High"], low=df["Low"], close=df["Close"], window=high)
        df = df.copy()
        df["don_high"] = donchian.donchian_channel_hband()
        df["don_mid"] = donchian.donchian_channel_mband()
        df["don_low"] = donchian.donchian_channel_lband()
    except ImportError:
        # Fallback: calculate manually
        df = df.copy()
        df["don_high"] = df["High"].rolling(window=high).max()
        df["don_low"] = df["Low"].rolling(window=low).min()
        df["don_mid"] = (df["don_high"] + df["don_low"]) / 2

    # Add Donchian Channels
    don_high_trace = go.Scatter(x=df.index, y=df["don_high"], name="don_high", line=dict(color="red", width=2, dash="dash"))
    fig.add_trace(don_high_trace, row=1, col=1)
    don_mid_trace = go.Scatter(x=df.index, y=df["don_mid"], name="don_mid", line=dict(color="blue", width=2, dash="dash"))
    fig.add_trace(don_mid_trace, row=1, col=1)
    don_low_trace = go.Scatter(x=df.index, y=df["don_low"], name="don_low", line=dict(color="purple", width=2, dash="dash"))
    fig.add_trace(don_low_trace, row=1, col=1)



def plot_strategy_multiple(market, tickers, data, out_dir, max_bars=200):
    """
    Plot multiple strategy charts for given tickers and data, and save the charts to the specified output directory.

    Parameters:
    - tickers: list of strings, tickers to plot
    - data: dictionary, key-value pairs where key is ticker and value is pandas DataFrame of stock data
    - out_dir: string, output directory to save the charts
    - max_bars: int, optional, maximum number of bars to include in the charts (default is 200)

    Returns:
    None
    """
    for i, ticker in enumerate(tickers):
        i += 1
        # get trading view recommendations
        recommendation = tools.tradingview_recommendation(ticker, market)
        recommendation_str = ' '.join(recommendation)
        # get the data for the ticker
        df = data[ticker]
        # removing all empty dates
        # build complete timeline from start date to end date
        dt_all = pd.date_range(start=df.index[0], end=df.index[-1])
        # retrieve the dates that are in the original datset
        dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
        # define dates with missing values
        dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
        df = df.tail(max_bars).copy()
        # Construct a 4 x 1 Plotly figure
        fig = make_subplots(
            rows=4,
            cols=1,
            vertical_spacing=0.01,
            shared_xaxes=True,
            row_heights=(2, 0.5, 1, 1),
        )

        # Plot the Price, SMA and EMA chart
        for col in ["Close", "SMA_10", "SMA_30", "SMA_50", "SMA_200"]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col), row=1, col=1)

        # Add the donchain chart
        add_donchain_chart(fig, df)

        # Add the volume chart
        add_volume_chart(fig, df)

        # Add the macd chart
        add_macd_chart(fig, df)

        # Add RSI chart
        add_rsi_chart(fig, df)

        # Adds the range selector
        add_range_selector(fig)

        # Set the color from white to black on range selector buttons
        fig.update_layout(xaxis=dict(rangeselector=dict(font=dict(color="black"))))

        # Add labels to y axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=4, col=1)

        # Sets customized padding
        set_padding(fig)

        # Remove dates without values
        fig.update_xaxes(tickangle=45, rangebreaks=[dict(values=dt_breaks)])

        # Set the template and the title
        layout = go.Layout(
            template="seaborn",
            title = str(i) + "_" + ticker + " - " + recommendation_str,
            height=900,
            width=1200,
            legend_title="Legend",
        )
        fig.update_layout(layout)
        pio.write_image(
            fig, f"{out_dir}/{i}_{ticker}.svg", format="svg"
        )


def create_plot_and_email(screener: str, market: str, symbols: list, data: dict, out_dir: str):
    """
    Generate an indicator plot for the given screener, market, symbols, data, and output directory.

    Args:
        screener (str): The name of the screener.
        market (str): The market for the symbols.
        symbols (list): List of symbols to generate indicator plots for.
        data (dict): Dictionary containing the price data for the symbols.
        out_dir (str): The output directory to save the generated plots.

    Returns:
        None
    """
    plot_data = {symbol: data["price_data"][symbol] for symbol in symbols}
    os.makedirs(out_dir, exist_ok=True)
    plot_strategy_multiple(market, symbols, plot_data, out_dir)
    send_email(market, screener, out_dir)



def create_plot_and_email_batched(screener: str, market: str, symbols: list, data: dict, out_dir: str, batch_size: int = 100):
    """
    Generate an indicator plot for the given screener, market, symbols, data, and output directory,
    and send email with a batch of symbols.

    Args:
        screener (str): The name of the screener.
        market (str): The market for the symbols.
        symbols (list): List of symbols to generate indicator plots for.
        data (dict): Dictionary containing the price data for the symbols.
        out_dir (str): The output directory to save the generated plots.
        batch_size (int, optional): The batch size for sending email. Defaults to 100.

    Returns:
        None
    """
    num_batches = math.ceil(len(symbols) / batch_size)
    for i in range(num_batches):
        batch_symbols = symbols[i * batch_size: (i + 1) * batch_size]
        plot_data = {symbol: data["price_data"][symbol] for symbol in batch_symbols}
        batch_out_dir = f"{out_dir}_batch_{i}"
        os.makedirs(batch_out_dir, exist_ok=True)
        plot_strategy_multiple(market, batch_symbols, plot_data, batch_out_dir)
        try:
            send_email(market, screener, batch_out_dir)
        except FileNotFoundError:
            # Email config not found, skip email sending
            pass