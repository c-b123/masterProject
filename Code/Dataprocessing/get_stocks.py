import yfinance as yf


# give dataframe with news articles get stocks
# using a window size of 7 
# use the most common 50 companies 
# keep track of dates and companies for which we have stock prices

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


company_tickers = [
    'MRK', 'MS', 'MU', 'NVDA', 'EBAY', 'NFLX', 'QQQ', 'VZ', 'DAL',
    'JNJ', 'QCOM', 'PFE', 'GSK', 'SNY', 'MRK.DE', 'GS', 'JPM', 'UBS', 'WFC',
    '000660.KS', 'INTC', 'WDC', 'AMD', 'MSFT', '2357.TW', 'AMZN', 'WMT', 'TGT',
    'HM-B.ST', 'DIS', 'AAPL', 'DJIA', 'SPX', 'IXIC', 'RUT', 'T', 'USM', 'LUMN',
    'FYBR', 'LUV', 'KLM', 'JBLU', 'LHA.DE', 'NVS', 'ABT', 'MDTKF'
]

# company_ticker = "AAPL"  # Apple Inc. stock ticker
# company_name = "Apple"  # Company name to search news
start_date = "2010-01-01"  # Start date for news and stock data
end_date = "2023-04-18"  # End date for news and stock data

for ticker in company_tickers:
    try:
        stock_data = get_stock_data(ticker, start_date, end_date)
        stock_data.to_csv(ticker + '.csv')
    except Exception as e:
        print(e)
        print(ticker)
