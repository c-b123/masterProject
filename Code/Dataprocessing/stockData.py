import yfinance as yf
import resources as r


def get_stock_data(ticker, start_date, end_date):
    """
    Returns a dataframe with date, open, high, low, close, adjusted close, and volume for a single stock.

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
    object
        dataframe with date, open, high, low, close, adjusted close, and volume as columns
    """

    stock_data = yf.download(ticker, start=start_date, end=end_date)

    return stock_data

# company_ticker = "AAPL"  # Apple Inc. stock ticker
# company_name = "Apple"  # Company name to search news
start_date = "2010-01-01"  # Start date for news and stock data
end_date = "2023-04-18"  # End date for news and stock data

for ticker in r.company_tickers:
    try:
        stock_data = get_stock_data(ticker, start_date, end_date)
        stock_data.to_csv(ticker + '.csv')
    except Exception as e:
        print(e)
        print(ticker)
