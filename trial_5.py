from advanced_ta import LorentzianClassification
import talib as ta
import yfinance as yf

result = {}

msft = yf.Ticker("AAPL")
df_hist = msft.history(period="3y",)
df_hist.index = df_hist.index.strftime('%Y-%m-%d')
df_hist['Adj Close'] = df_hist['Close'].copy()
df = df_hist.reindex(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits'])

# df here is the dataframe containing stock data as [['open', 'high', 'low', 'close', 'volume']]. Notice that the column names are in lower case.
lc = LorentzianClassification(
    df,
    features=[
        LorentzianClassification.Feature("RSI", 14, 2),  # f1
        LorentzianClassification.Feature("WT", 10, 11),  # f2
        LorentzianClassification.Feature("CCI", 20, 2),  # f3
        LorentzianClassification.Feature("ADX", 20, 2),  # f4
        LorentzianClassification.Feature("RSI", 9, 2),   # f5
        ta.MFI(df['open'], df['high'], df['low'], df['close'], df['volume']) #f6
    ],
    settings=LorentzianClassification.Settings(
        source='close',
        neighborsCount=8,
        maxBarsBack=2000,
        useDynamicExits=False
    ),
    filterSettings=LorentzianClassification.FilterSettings(
        useVolatilityFilter=True,
        useRegimeFilter=True,
        useAdxFilter=False,
        regimeThreshold=-0.1,
        adxThreshold=20,
        kernelFilter = LorentzianClassification.KernelFilter(
            useKernelSmoothing = False,
            lookbackWindow = 8,
            relativeWeight = 8.0,
            regressionLevel = 25,
            crossoverLag = 2,
        )
    ))
lc.dump('output/result.csv')
lc.plot('output/result.jpg')

#####################################################################################################################

def volatile_data():
        si_columns = ["SYMBOL", "CURRENCY", "SECTOR", "INDUSTRY"]
    if market == "india":
        si_filename = "data/historic_data/india/stock_info.csv"
    else:
        si_filename = "data/historic_data/us/stock_info.csv"
    if not os.path.exists(si_filename):
        # create a .csv to store stock information
        with open(si_filename, "w") as file:
            wr = csv.writer(file)
            wr.writerow(si_columns)
    # load stock information file
    si = pd.read_csv(si_filename)
    missing_tickers = [
        ticker for ticker in tickers if ticker not in si["SYMBOL"].values
    ]
    missing_si = {}
    currencies = {}

                if ticker in missing_tickers:
                if market == "india":
                    currencies[ticker] = "INR"
                else:
                    currencies[ticker] = "USD"
                try:
                    assert (
                        (len(company_info_one["sector"]) > 0)
                        and (len(company_info_one["industry"]) > 0)
                    )
                    missing_si[ticker] = dict(
                        sector=company_info_one["sector"], industry=company_info_one["industry"]
                    )


    info = zip(
        list(missing_si.keys()),
        [currencies[ticker] for ticker in missing_si.keys()],
        [v["sector"] for v in missing_si.values()],
        [v["industry"] for v in missing_si.values()],
    )
    with open(si_filename, "a+", newline="") as file:
        wr = csv.writer(file)
        for row in info:
            wr.writerow(row)
    si = pd.read_csv(si_filename).set_index("SYMBOL").to_dict(orient="index")

    missing_tickers = [
        ticker
        for ticker in tickers
        if ticker not in data.columns.get_level_values(0)[::2].tolist()
    ]
    tickers = data.columns.get_level_values(0)[::2].tolist()
    if len(missing_tickers) > 0:
        print(
            "\nRemoving {} from list of symbols because we could not collect full information.".format(
                missing_tickers
            )
        )

    # download exchange rates and convert to most common currency
    currencies = [
        si[ticker]["CURRENCY"] if ticker in si else currencies[ticker]
        for ticker in tickers
    ]
    ucurrencies, counts = np.unique(currencies, return_counts=True)
    default_currency = ucurrencies[np.argmax(counts)]
    xrates = get_exchange_rates(
        currencies, default_currency, data.index, start, end, interval
    )

    return dict(
        tickers=tickers,
        dates=pd.to_datetime(data.index),
        price=data.iloc[:, data.columns.get_level_values(1) == "Adj Close"]
        .to_numpy()
        .T,
        volume=data.iloc[:, data.columns.get_level_values(1) == "Volume"].to_numpy().T,
        currencies=currencies,
        exchange_rates=xrates,
        default_currency=default_currency,
        sectors={
            ticker: si[ticker]["SECTOR"] if ticker in si else "NA_" + ticker
            for ticker in tickers
        },
        industries={
            ticker: si[ticker]["INDUSTRY"] if ticker in si else "NA_" + ticker
            for ticker in tickers
        },
        price_data=price_data,
        company_info=company_info,
    )