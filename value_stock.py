import os
import requests
import pandas as pd
from io import StringIO
from src.classes.Send_email import send_email, send_email_volatile

def main():
    market = "india"
    # Start processing data
    if not os.path.exists("data/processed_data/volatile"):
        os.makedirs("data/processed_data/volatile", exist_ok=True)
    screener_volatile_dir = "data/processed_data/volatile"
    print("\nStarting Volatility based screening...")
    send_email_volatile(market, "Volatility", screener_volatile_dir)
    print("\nFinished Volatility based screening...")

if __name__ == "__main__":
    main()