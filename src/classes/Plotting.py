import os
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
# For plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


def compute_uncertainty_bounds(est: np.array, std: np.array):
    return np.maximum(0, est - 2 * std), est + 2 * std


def plot_market_estimates(data: dict, est: np.array, std: np.array):
    """
    It makes a market estimation plot with prices, trends, uncertainties and volumes.

    Parameters
    ----------
    data: dict
        Downloaded data.
    est: np.array
        Price trend estimate at market-level.
    std: np.array
        Standard deviation estimate of price trend at market-level.
    """
    print('\nPlotting market estimation...')
    fig = plt.figure(figsize=(10, 3))
    logp = np.log(data['price'])
    t = logp.shape[1]
    lb, ub = compute_uncertainty_bounds(est, std)

    plt.grid(axis='both')
    plt.title("Market", fontsize=15)
    avg_price = np.exp(logp.mean(0))
    l1 = plt.plot(data["dates"], avg_price, label="avg. price in {}".format(data['default_currency']), color="C0")
    l2 = plt.plot(data["dates"], est[0], label="trend", color="C1")
    l3 = plt.fill_between(data["dates"], lb[0], ub[0], alpha=0.2, label="+/- 2 st. dev.", color="C0")
    plt.ylabel("avg. price in {}".format(data['default_currency']), fontsize=12)
    plt.twinx()
    l4 = plt.bar(data["dates"], data['volume'].mean(0), width=1, color='g', alpha=0.2, label='avg. volume')
    l4[0].set_edgecolor('r')
    for d in range(1, t):
        if avg_price[d] - avg_price[d - 1] < 0:
            l4[d].set_color('r')
    plt.ylabel("avg. volume", fontsize=12)
    ll = l1 + l2 + [l3] + [l4]
    labels = [l.get_label() for l in ll]
    plt.legend(ll, labels, loc="upper left")

    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig_name = 'plots/market_estimation.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    print('Market estimation plot has been saved to {}/{}.'.format(os.getcwd(), fig_name))


def plot_sector_estimates(data: dict, info: dict, est: np.array, std: np.array):
    """
    It makes a plot for each sector with prices, trends, uncertainties and volumes.

    Parameters
    ----------
    data: dict
        Downloaded data.
    info: dict
        Model hierarchy information.
    est: np.array
        Price trend estimate at sector-level.
    std: np.array
        Standard deviation estimate of price trend at sector-level.
    """
    print('\nPlotting sector estimation...')
    num_columns = 3
    logp = np.log(data['price'])
    t = logp.shape[1]
    lb, ub = compute_uncertainty_bounds(est, std)

    NA_sectors = np.where(np.array([sec[:2] for sec in info['unique_sectors']]) == "NA")[0]
    num_NA_sectors = len(NA_sectors)
    
    fig = plt.figure(figsize=(20, max(info['num_sectors'] - num_NA_sectors, 5)))
    j = 0
    for i in range(info['num_sectors']):
        if i not in NA_sectors:
            j += 1
            plt.subplot(int(np.ceil((info['num_sectors'] - num_NA_sectors) / num_columns)), num_columns, j)
            plt.grid(axis='both')
            plt.title(info['unique_sectors'][i], fontsize=15)
            idx_sectors = np.where(np.array(info['sectors_id']) == i)[0]
            avg_price = np.exp(logp[idx_sectors].reshape(-1, t).mean(0))
            l1 = plt.plot(data["dates"], avg_price,
                          label="avg. price in {}".format(data['default_currency']), color="C0")
            l2 = plt.plot(data["dates"], est[i], label="trend", color="C1")
            l3 = plt.fill_between(data["dates"], lb[i], ub[i], alpha=0.2, label="+/- 2 st. dev.",
                                  color="C0")
            plt.ylabel("avg. price in {}".format(data['default_currency']), fontsize=12)
            plt.xticks(rotation=45)
            plt.twinx()
            l4 = plt.bar(data["dates"],
                         data['volume'][np.where(np.array(info['sectors_id']) == i)[0]].reshape(-1, t).mean(0),
                         width=1, color='g', alpha=0.2, label='avg. volume')
            for d in range(1, t):
                if avg_price[d] - avg_price[d - 1] < 0:
                    l4[d].set_color('r')
            l4[0].set_edgecolor('r')
            plt.ylabel("avg. volume", fontsize=12)
            ll = l1 + l2 + [l3] + [l4]
            labels = [l.get_label() for l in ll]
            plt.legend(ll, labels, loc="upper left")
    plt.tight_layout()

    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig_name = 'plots/sector_estimation.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    print('Sector estimation plot has been saved to {}/{}.'.format(os.getcwd(), fig_name))


def plot_industry_estimates(data: dict, info: dict, est: np.array, std: np.array):
    """
    It makes a plot for each industry with prices, trends, uncertainties and volumes.

    Parameters
    ----------
    data: dict
        Downloaded data.
    info: dict
        Model hierarchy information.
    est: np.array
        Price trend estimate at industry-level.
    std: np.array
        Standard deviation estimate of price trend at industry-level.
    """
    print('\nPlotting industry estimation...')
    num_columns = 3
    logp = np.log(data['price'])
    t = logp.shape[1]
    lb, ub = compute_uncertainty_bounds(est, std)

    NA_industries = np.where(np.array([ind[:2] for ind in info['unique_industries']]) == "NA")[0]
    num_NA_industries = len(NA_industries)

    fig = plt.figure(figsize=(20, max(info['num_industries'] - num_NA_industries, 5)))
    j = 0
    for i in range(info['num_industries']):
        if i not in NA_industries:
            j += 1
            plt.subplot(int(np.ceil((info['num_industries'] - num_NA_industries) / num_columns)), num_columns, j)
            plt.grid(axis='both')
            plt.title(info['unique_industries'][i], fontsize=15)
            idx_industries = np.where(np.array(info['industries_id']) == i)[0]
            plt.title(info['unique_industries'][i], fontsize=15)
            avg_price = np.exp(logp[idx_industries].reshape(-1, t).mean(0))
            l1 = plt.plot(data["dates"], avg_price,
                          label="avg. price in {}".format(data['default_currency']), color="C0")
            l2 = plt.plot(data["dates"], est[i], label="trend", color="C1")
            l3 = plt.fill_between(data["dates"], lb[i], ub[i], alpha=0.2, label="+/- 2 st. dev.",
                                  color="C0")
            plt.ylabel("avg. price in {}".format(data['default_currency']), fontsize=12)
            plt.xticks(rotation=45)
            plt.twinx()
            l4 = plt.bar(data["dates"],
                         data['volume'][np.where(np.array(info['industries_id']) == i)[0]].reshape(-1, t).mean(0),
                         width=1, color='g', alpha=0.2, label='avg. volume')
            for d in range(1, t):
                if avg_price[d] - avg_price[d - 1] < 0:
                    l4[d].set_color('r')
            l4[0].set_edgecolor('r')
            plt.ylabel("avg. volume", fontsize=12)
            ll = l1 + l2 + [l3] + [l4]
            labels = [l.get_label() for l in ll]
            plt.legend(ll, labels, loc="upper left")
    plt.tight_layout()

    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig_name = 'plots/industry_estimation.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    print('Industry estimation plot has been saved to {}/{}.'.format(os.getcwd(), fig_name))


def plot_stock_estimates(data: dict, est: np.array, std: np.array, rank_type: str, rank: list, ranked_rates: np.array):
    """
    It makes a plot for each stock with prices, trends, uncertainties and volumes.

    Parameters
    ----------
    data: dict
        Downloaded data.
    est: np.array
        Price trend estimate at stock-level.
    std: np.array
        Standard deviation estimate of price trend at stock-level.
    rank_type: str
        Type of rank. It can be either `rate` or `growth`.
    rank: list
        List of integers at stock-level indicating the rank specified in `rank_type`.
    ranked_rates: np.array
        Array of rates at stock-level ranked according to `rank`.
    """
    num_stocks, t = data['price'].shape

    # determine which stocks are along trend to avoid plotting them
    if rank_type == "rate":
        to_plot = np.where(np.array(ranked_rates) != "ALONG TREND")[0]
    elif rank_type == "growth":
        to_plot = np.where(np.array(ranked_rates) == "ALONG TREND")[0][:99]
    elif rank_type == "volatility":
        to_plot = np.arange(99)
    num_to_plot = len(to_plot)

    if num_to_plot > 0:
        print('\nPlotting stock estimation...')
        num_columns = 3

        ranked_tickers = np.array(data['tickers'])[rank]
        ranked_p = data['price'][rank]
        ranked_volume = data['volume'][rank]
        ranked_currencies = np.array(data['currencies'])[rank]
        ranked_est = est[rank]
        ranked_std = std[rank]

        ranked_lb, ranked_ub = compute_uncertainty_bounds(ranked_est, ranked_std)

        j = 0
        fig = plt.figure(figsize=(20, max(num_to_plot, 5)))
        for i in range(num_stocks):
            if i in to_plot:
                j += 1
                plt.subplot(int(np.ceil(num_to_plot / num_columns)), num_columns, j)
                plt.grid(axis='both')
                plt.title(ranked_tickers[i], fontsize=15)
                l1 = plt.plot(data["dates"], ranked_p[i], label="price in {}".format(ranked_currencies[i]))
                l2 = plt.plot(data["dates"], ranked_est[i], label="trend")
                l3 = plt.fill_between(data["dates"], ranked_lb[i], ranked_ub[i], alpha=0.2,
                                      label="+/- 2 st. dev.")
                plt.yticks(fontsize=12)
                plt.xticks(rotation=45)
                plt.ylabel("price in {}".format(ranked_currencies[i]), fontsize=12)
                plt.twinx()
                l4 = plt.bar(data["dates"], ranked_volume[i], width=1, color='g', alpha=0.2, label='volume')
                for d in range(1, t):
                    if ranked_p[i, d] - ranked_p[i, d - 1] < 0:
                        l4[d].set_color('r')
                l4[0].set_edgecolor('r')
                plt.ylabel("volume", fontsize=12)
                ll = l1 + l2 + [l3] + [l4]
                labels = [l.get_label() for l in ll]
                plt.legend(ll, labels, loc="upper left")
        plt.tight_layout()

        if not os.path.exists('plots'):
            os.mkdir('plots')
        fig_name = 'plots/stock_estimation.png'
        fig.savefig(fig_name, dpi=fig.dpi)
        print('Stock estimation plot has been saved to {}/{}.'.format(os.getcwd(), fig_name))

    elif os.path.exists('plots/stock_estimation.png'):
        os.remove('plots/stock_estimation.png')


def plot_matches(data: dict, matches: dict):
    """
    It plots matches with smallest computed distance.

    Parameters
    ----------
    data: dict
        Downloaded data.
    matches: dict
        For each symbol, this dictionary contains a corresponding `match` symbol, the `index` of the match symbol in the
        list of symbols and the computed `distance` between the two.
    """
    print('\nPlotting matches estimation...')
    num_columns = 3

    tickers = np.array(data['tickers'])
    num_to_plot = min(len(tickers), 99)
    prices = data['price']
    currencies = np.array(data['currencies'])

    idx = np.argsort([matches[ticker]['distance'] for ticker in tickers])
    matched_idx = np.unique([{i, matches[tickers[i]]['index']} for i in idx]).tolist()[:num_to_plot]

    fig = plt.figure(figsize=(20, max(num_to_plot, 5)))
    for j, couple in enumerate(matched_idx):
        i1, i2 = tuple(couple)
        ticker, match = tickers[i1], tickers[i2]
        plt.subplot(int(np.ceil(num_to_plot / num_columns)), num_columns, j + 1)
        plt.grid(axis='both')
        plt.title("{} & {}".format(ticker, match), fontsize=15)

        l1 = plt.plot(data["dates"], prices[i1], c="C0", label="price of {} in {}".format(tickers[i1], currencies[i1]))
        plt.ylabel("price of {} in {}".format(tickers[i1], currencies[i1]), fontsize=12)
        plt.twinx()
        l2 = plt.plot(data["dates"], prices[i2], c="C1", label="price of {} in {}".format(match, currencies[i2]))
        plt.ylabel("price of {} in {}".format(match, currencies[i2]), fontsize=12)
        ll = l1 + l2
        labels = [l.get_label() for l in ll]
        plt.legend(ll, labels, loc="upper left")
    plt.tight_layout()

    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig_name = 'plots/matches_estimation.png'
    print('Matches estimation plot has been saved to {}/{}.'.format(os.getcwd(), fig_name))
    fig.savefig(fig_name, dpi=fig.dpi)


def plot_stocks_set_exploration(data, est, std, idx_set, num_rows=3, num_cols=3):
    tickers = data['tickers']
    p = data['price']
    currencies = data['currencies']
    volume = data['volume']
    lb, ub = compute_uncertainty_bounds(est, std)
    t = p.shape[1]

    plt.figure(figsize=(18, 7))
    for i, idx in enumerate(idx_set):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.grid(axis='both')
        plt.title(tickers[idx], fontsize=15)
        l1 = plt.plot(data["dates"], p[idx], label="price in {}".format(currencies[idx]))
        plt.yticks(fontsize=12)
        plt.xticks(rotation=45)
        plt.ylabel("price in {}".format(currencies[idx]), fontsize=12)
        l2 = plt.plot(data["dates"], est[idx], label="trend")
        l3 = plt.fill_between(data["dates"], lb[idx], ub[idx], alpha=0.2, label="+/- 2 st. dev.")
        plt.yticks(fontsize=12)
        plt.xticks(rotation=45)
        plt.ylabel("price in {}".format(currencies[idx]), fontsize=12)
        plt.twinx()
        l4 = plt.bar(data["dates"], volume[idx], width=1, color='g', alpha=0.2, label='volume')
        for d in range(1, t):
            if p[idx, d] - p[idx, d - 1] < 0:
                l4[d].set_color('r')
        l4[0].set_edgecolor('r')
        plt.ylabel("volume", fontsize=12)
        ll = l1 + l2 + [l3] + [l4]
        labels = [l.get_label() for l in ll]
        plt.legend(ll, labels, loc="upper left")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=False)


def plot_chosen_stocks_exploration(data, est, std, idx_choice_all, num_cols=3):
    tickers = data['tickers']
    p = data['price']
    currencies = data['currencies']
    volume = data['volume']
    lb, ub = compute_uncertainty_bounds(est, std)
    t = p.shape[1]

    num_to_plot = len(idx_choice_all)
    fig = plt.figure(figsize=(20, max(num_to_plot, 5)))
    for i, idx in enumerate(idx_choice_all):
        plt.subplot(int(np.ceil(num_to_plot / num_cols)), num_cols, i+1)
        plt.grid(axis='both')
        plt.title(tickers[idx], fontsize=15)
        l1 = plt.plot(data["dates"], p[idx], label="price in {}".format(currencies[idx]))
        plt.yticks(fontsize=12)
        plt.xticks(rotation=45)
        plt.ylabel("price in {}".format(currencies[idx]), fontsize=12)
        l2 = plt.plot(data["dates"], est[idx], label="trend")
        l3 = plt.fill_between(data["dates"], lb[idx], ub[idx], alpha=0.2, label="+/- 2 st. dev.")
        plt.yticks(fontsize=12)
        plt.xticks(rotation=45)
        plt.ylabel("price in {}".format(currencies[idx]), fontsize=12)
        plt.twinx()
        l4 = plt.bar(data["dates"], volume[idx], width=1, color='g', alpha=0.2, label='volume')
        for d in range(1, t):
            if p[idx, d] - p[idx, d - 1] < 0:
                l4[d].set_color('r')
        l4[0].set_edgecolor('r')
        plt.ylabel("volume", fontsize=12)
        ll = l1 + l2 + [l3] + [l4]
        labels = [l.get_label() for l in ll]
        plt.legend(ll, labels, loc="upper left")
    plt.tight_layout()

    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig_name = 'plots/exploration_chosen_stocks.png'
    print('Plot of the stocks chosen during the exploration has been saved to {}/{}.'.format(os.getcwd(), fig_name))
    fig.savefig(fig_name, dpi=fig.dpi)

# Sets padding for figures
def set_padding(fig):
    fig.update_layout(margin=go.layout.Margin(
        r=10, #right margin
        b=10)) #bottom margin

# Adds the range selector to given figure
def add_range_selector(fig):
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ]),
            type='date'),#end xaxis  definition
        xaxis2_type='date')

# Adds the volume chart to row 2, column 1
def add_volume_chart(fig, df):
    # Colours for the Bar chart
    colors = ['#9C1F0B' if row['Open'] - row['Close'] >= 0
          else '#2B8308' for index, row in df.iterrows()]

    # Adds the volume as a bar chart
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], showlegend=False, marker_color=colors), row=2, col=1)

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

def plot_strategy_multiple(tickers, data, max_bars=200):
    for i, ticker in enumerate(tickers):
        df = data[ticker]
        # removing all empty dates
        # build complete timeline from start date to end date
        dt_all = pd.date_range(start=df.index[0],end=df.index[-1])
        # retrieve the dates that are in the original datset
        dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
        # define dates with missing values
        dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
        df = df.tail(max_bars)
        # Construct a 2 x 1 Plotly figure
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.01, shared_xaxes=True)

        # Plot the Price, SMA and EMA chart
        for col in ['Close', 'SMA_10', 'SMA_30', 'SMA_50', 'SMA_200']:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col), row=1, col=1)

        # Add the volume chart
        add_volume_chart(fig, df)

        # Adds the range selector
        add_range_selector(fig)

        # Set the color from white to black on range selector buttons
        #fig.update_layout(xaxis=dict(rangeselector = dict(font = dict( color = 'black'))))

        # Add labels to y axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Sets customized padding
        set_padding(fig)

        # Remove dates without values
        fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

        # Set the template and the title
        layout = go.Layout(template="seaborn", title = ticker + ' - Price and Volume', height=500, legend_title='Legend')
        fig.update_layout(layout)
        pio.write_image(fig, f"plots/{ticker}.png")
