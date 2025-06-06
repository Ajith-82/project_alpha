import pandas as pd


def get_value_screener_data(statistics, balance_sheet, income_statement):
    '''
    Find stocks that meet the criteria for value screener.
    '''
    ticker_data = pd.Series()
    # TODO Get EPS from statistics

    ticker_data["EPS"] = eps
    # TODO Get Debt to equity from balance sheet
    ticker_data["D_E"] = debt_to_equity

    # TODO Get Average return on capital employed from income statement
    try:
        capital_employed = balance_sheet.loc['Total Assets'] - balance_sheet.loc['Total Debt']
        roce = (income_statement.loc['Net Income'] / capital_employed).tolist()
        # calculate average of list of roce
        aroce = sum(roce) / len(roce)
        ticker_data["AROCE"] = aroce
    except Exception as e:
        print(e)
    # TODO Get Market Capitalization from statistics
    ticker_data["MCAP"] = mcap

    # TODO Get OPM from income statement
    ticker_data["OPM"] = opm
    pass

def find_value_stocks(filtered_data):
    '''
    Find stocks that meet the criteria for value screener.
    '''
    pass

def value_screener(data, tickers):
    '''
    Generate a value based screener for a given set of stocks based on various signal indicators.
    EPS last year >20 AND
    Debt to equity <.1 AND
    Average return on capital employed 5Years >35 AND
    Market Capitalization >500 AND
    OPM 5Year >15
    
    Args:
        data (dict): A dictionary containing "tickers" and "price_data".
        duration (int): The duration in days for which the MACD signals are generated.
        
    Returns:
        dict: A dictionary containing the matching trades for each stock based on the specified conditions in the last `duration` days.
    '''

    statistics = data["company_info"]
    balance_sheet = data["balance_sheet"]
    income_statement = data["income_statement"]

    screener_data = []

    for ticker in tickers:
        value_stock = find_value_stocks(statistics, balance_sheet, income_statement)
        if value_stock:
            screener_data.append(ticker)
    return screener_data