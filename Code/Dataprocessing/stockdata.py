import yfinance as yf
import pandas as pd
import numpy as np


def get_stock_data(ticker, start_date, end_date):
    """
    Fetches open, high, low, close, adjusted close and volume for a single stock. Subsequently, return and log return is
    calculated.

    Parameters
    ----------
    ticker : str
        The company ticker.
    start_date : str
        The start date in the format: %y-%m-%d.
    end_date : str
        The end date in the format: %y-%m-%d.

    Returns
    -------
    pandas.DataFrame
        dataframe with date, open, high, low, close, adjusted close, volume, return, and log return as columns
    """

    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        # Create date column
        stock_data.reset_index(inplace=True)
        stock_data = stock_data.rename(columns={'index': 'date'})

        # Lower case to allow merging on data and ticker
        stock_data.rename(str.lower, axis='columns', inplace=True)

        # Convert from datetime to date format
        stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date

        # Convert "date" column to type str to allow merging
        stock_data['date'].astype(str)

        # Calculate return and log return
        stock_data["return"] = stock_data["adj close"].pct_change()
        stock_data["log_return"] = np.log(1 + stock_data["return"])

        # Drop nan
        stock_data.dropna(inplace=True, ignore_index=True)

        return stock_data

    except Exception as e:
        print(f"Cannot fetch data for: {ticker}. Error: {e}")
