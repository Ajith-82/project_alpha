#!/usr/bin/env python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd


# Sets padding for figures
def set_padding(fig):
    fig.update_layout(
        margin=go.layout.Margin(r=10, b=10)  # right margin
    )  # bottom margin


# Adds the range selector to given figure
def add_range_selector(fig):
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
            type="date",
        ),  # end xaxis  definition
        xaxis2_type="date",
    )


# Adds the volume chart to row 2, column 1
def add_volume_chart(fig, df):
    # Colours for the Bar chart
    colors = [
        "#9C1F0B" if row["Open"] - row["Close"] >= 0 else "#2B8308"
        for index, row in df.iterrows()
    ]

    # Adds the volume as a bar chart
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], showlegend=False, marker_color=colors),
        row=2,
        col=1,
    )


# Adds the MACD chart to row 3, column 1
def add_macd_chart(fig, df):
    # Colours for the Bar chart
    colors = [
        "green" if row["MACD_hist"] >= 0 else "red" for index, row in df.iterrows()
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["MACD_hist"], showlegend=False, marker_color=colors),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["MACD"], name="MACD", line=dict(color="black", width=2)
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_signal"],
            name="MACD_signal",
            line=dict(color="blue", width=1),
        ),
        row=3,
        col=1,
    )


# Adds the RSI chart to row 4, column 1
def add_rsi_chart(fig, df):
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple", width=2)
        ),
        row=4,
        col=1,
    )
    # Add horizontal lines for Y=30 and Y=70
    fig.add_shape(
        dict(
            type="line",
            x0=df.index[0],
            x1=df.index[-1],
            y0=30,
            y1=30,
            line=dict(color="green", width=1),
        ),
        row=4,
        col=1,
    )

    fig.add_shape(
        dict(
            type="line",
            x0=df.index[0],
            x1=df.index[-1],
            y0=70,
            y1=70,
            line=dict(color="green", width=1),
        ),
        row=4,
        col=1,
    )
    fig.update_yaxes(range=[0, 100], row=4, col=1)


def plot_strategy_multiple(tickers, data, out_dir, max_bars=200):
    for i, ticker in enumerate(tickers):
        df = data[ticker]
        # removing all empty dates
        # build complete timeline from start date to end date
        dt_all = pd.date_range(start=df.index[0], end=df.index[-1])
        # retrieve the dates that are in the original datset
        dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
        # define dates with missing values
        dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
        df = df.tail(max_bars)
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
            title=ticker + " - Price and Indicators",
            height=900,
            width=1200,
            legend_title="Legend",
        )
        fig.update_layout(layout)
        pio.write_image(
            fig, f"{out_dir}/{ticker}.svg", format="svg"
        )
