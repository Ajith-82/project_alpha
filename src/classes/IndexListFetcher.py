import requests
import tempfile
import pandas as pd
from bs4 import BeautifulSoup

# class IndexListFetcher:
#    def __init__(self):
#        pass


# Write to CSV
def _write_to_csv_file(symbols_list):
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv"
    ) as temp_file:
        file_path = temp_file.name
        symbols_list.to_csv(file_path, index=False)
    return file_path


def split_security_name(name):
    if pd.notna(name):  # Check if the value is not NaN
        return name.split("-")[
            0
        ].strip()  # Split and take the first part, stripping any extra spaces
    else:
        return None


def _clean_data(df):
    df = df.copy()
    # Remove test listings
    df = df[df["Test Issue"] == "N"]

    # Create New Column w/ Just Company Name
    df["Company Name"] = df["Security Name"].apply(split_security_name)

    # Move Company Name to 2nd Col
    cols = list(df.columns)
    cols.insert(1, cols.pop(-1))
    df = df.loc[:, cols]
    return df


def _screener_clean_data(df):
    # Remove unwanted rows
    df = df.dropna()
    df = df[df["Name"] != "Name"]

    # Reset dataframe index
    df.reset_index(drop=True, inplace=True)
    df["Symbol"] = df["Name"].apply(get_ticker)
    df["Symbol"] = df["Symbol"].str.replace(".NS", "", regex=False)
    df["Symbol"] = df["Symbol"].str.replace(".BO", "", regex=False)

    # Move Symbol to 3rd Col
    cols = list(df.columns)
    cols.insert(2, cols.pop(-1))
    df = df.loc[:, cols]
    return df


def _nse_all_stocks():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    df = df[["SYMBOL", "NAME OF COMPANY"]]
    new_columns = ["symbol", "Name"]
    df.columns = new_columns
    df["Name"] = df["Name"].str.lower()
    df["Name"] = df["Name"].str.replace(" limited", "")
    return df


def get_ticker(company_name):
    try:
        yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        params = {"q": company_name, "quotes_count": 1, "country": "India"}
        res = requests.get(
            url=yfinance, params=params, headers={"User-Agent": user_agent}
        )
        data = res.json()
        company_code = data["quotes"][0]["symbol"]
        return company_code
    except IndexError:
        pass

def _create_file_schema(df, filename):
    fields = []
    for name, dtype in zip(df.columns, df.dtypes):
        if (
            str(dtype) == "object" or str(dtype) == "boolean"
        ):  # does datapackage.json use boolean type?
            dtype = "string"
        else:
            dtype = "number"
        fields.append({"name": name, "description": "", "type": dtype})

    return {
        "name": filename,
        "path": filename + ".csv",
        "format": "csv",
        "mediatype": "text/csv",
        "schema": {"fields": fields},
    }


def _create_datapackage(datasets):
    resources = []
    for df, filename in datasets:
        resources.append(_create_file_schema(df, filename))

    return {
        "name": PACKAGE_NAME,
        "title": PACKAGE_TITLE,
        "license": "",
        "resources": resources,
    }


def nasdaq_symbols_list():
    symbols_file = (
        "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"  # Nasdaq only
    )
    symbols_raw = pd.read_csv(symbols_file, sep="|")
    symbols_raw = _clean_data(symbols_raw)
    symbols_filtered = symbols_raw[
        (symbols_raw["Market Category"] == "S")
        & (symbols_raw["Financial Status"] == "N")
        & (symbols_raw["Test Issue"] == "N")
    ]
    symbols_list = symbols_filtered["Symbol"].tolist()
    return symbols_list


def nyse_symbols_list():
    symbols_file = "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt"  # All other exchanges
    symbols_raw = pd.read_csv(symbols_file, sep="|")
    symbols_raw = _clean_data(symbols_raw)
    symbols_raw["Symbol"] = symbols_raw["ACT Symbol"]
    symbols_filtered = symbols_raw[
        (symbols_raw["Exchange"] == "N") & (symbols_raw["ETF"] == "N")
    ]
    symbols_list = symbols_filtered["Symbol"].tolist()
    return symbols_list

def screener_value():
    link1 = "https://www.screener.in/screens/184/value-stocks/?sort=price+to+earning&order=desc&limit=50&page=1"
    link2 = "https://www.screener.in/screens/184/value-stocks/?sort=price+to+earning&order=desc&limit=50&page=2"
    symbols_raw = pd.read_html(link1) + pd.read_html(link2)
    symbols_raw = pd.concat(symbols_raw, ignore_index=True)
    symbols_raw = _screener_clean_data(symbols_raw)
    symbols_list = symbols_raw["Symbol"].tolist()
    return symbols_list

def sp_500_old():
    link = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks"
    symbols_raw = pd.read_html(link, header=0)[0]
    symbols_list = symbols_raw["Symbol"].tolist()
    return symbols_list

# Data source www.stockmonitor.com
def get_sector_stocks_list(url):
    '''
    Function to fetch stock sector or stock list from a URL.
    
    Args:
        url (str): The URL to fetch data from.
    
    Returns:
        list: A list of stock sectors or lists.
    '''
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.text, 'html.parser')
        elements = [element.text for element in soup.select("tbody tr td.text-left a")]
        return elements
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

def us_sector_dict():
    # Data source URLs.
    sectors_url = "https://www.stockmonitor.com/sectors/"
    base_url = "https://www.stockmonitor.com/sector/"

    # Get sector list
    sector_list = get_sector_stocks_list(sectors_url)
    sector_stocks = {}

    for key in sector_list:
        if len(key.split(" ")) > 2:
            url_key = f"{key.split(' ')[0]}-{key.split(' ')[1]}".lower()
        else:
            url_key = f"{key.split(' ')[0]}".lower()

        sector_url = f"{base_url}{url_key}/"
        sector_stocks[key] = get_sector_stocks_list(sector_url)

    return sector_stocks

def sp_500():
    url = "https://www.stockmonitor.com/sp500-stocks/"
    return get_sector_stocks_list(url)

def tech_100():
    url = "https://www.stockmonitor.com/nasdaq-stocks/"
    return get_sector_stocks_list(url)

def dow_jones():
    url = "https://www.stockmonitor.com/dji-stocks/"
    return get_sector_stocks_list(url)

def sector_list(sector_name):
    # Data source URLs.
    base_url = "https://www.stockmonitor.com/sector/"

    sector_stocks = []

    if len(sector_name.split(" ")) > 2:
        url_key = f"{sector_name.split(' ')[0]}-{sector_name.split(' ')[1]}".lower()
    else:
        url_key = f"{sector_name.split(' ')[0]}".lower()

    sector_url = f"{base_url}{url_key}/"
    sector_stocks = get_sector_stocks_list(sector_url)

    return sector_stocks

def nse_500():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    df = pd.read_csv(url)
    return df["Symbol"].to_list()

def main():
    # screener_value()
    #print(us_sector_dict())
    print(nse_500())

if __name__ == "__main__":
    main()
    
