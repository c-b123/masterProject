import pandas as pd
import yfinance as yf
import yfinance.shared as shared
import resources as r


def calculate_statistics(symbols, start_date, end_date):
    # Initialize an empty DataFrame to store daily returns
    returns_df = pd.DataFrame()
    errors = []

    # Iterate over symbols and download historical stock data
    for symbol in symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        errors.extend(list(shared._ERRORS.keys()))

        # Calculate daily returns
        stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()

        # Reset index and rename 'Date' column
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={'Date': 'Date'}, inplace=True)

        # Remove rows with NaN values
        stock_data = stock_data.dropna()

        # Merge daily returns with returns_df based on date
        if returns_df.empty:
            returns_df = stock_data[['Date', 'Daily_Return']].copy()
            returns_df.rename(columns={'Daily_Return': symbol}, inplace=True)
            returns_df.set_index('Date', inplace=True)
        else:
            returns_df = returns_df.merge(stock_data[['Date', 'Daily_Return']], left_index=True, right_on='Date',
                                          how='left')
            returns_df.rename(columns={'Daily_Return': symbol}, inplace=True)
            returns_df.set_index('Date', inplace=True)

    # Calculate statistics on the returns_df DataFrame
    statistics = pd.DataFrame()
    statistics['Mean'] = returns_df.mean(axis=1)
    statistics['Variance'] = returns_df.var(axis=1)
    percentiles = [0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9]
    for percentile in percentiles:
        statistics[f'{int(percentile * 100)}th Percentile'] = returns_df.quantile(percentile, axis=1)

    return statistics, errors


# Define the date range
start_date = '2010-01-01'
end_date = '2020-12-31'

# Calculate the statistics
stats, errors = calculate_statistics(r.sp500, start_date, end_date)

# Create a DataFrame to store the results
result_df = pd.DataFrame({'Date': stats.index,
                          'Mean': stats['Mean'],
                          'Variance': stats['Variance'],
                          '10th_Percentile': stats['10th Percentile'],
                          '20th_Percentile': stats['10th Percentile'],
                          '25th_Percentile': stats['25th Percentile'],
                          '50th_Percentile': stats['50th Percentile'],
                          '75th_Percentile': stats['75th Percentile'],
                          '80th_Percentile': stats['75th Percentile'],
                          '90th_Percentile': stats['90th Percentile']})

# Lower all column names
result_df.rename(str.lower, axis='columns', inplace=True)

# Convert from datetime to date format
result_df['date'] = pd.to_datetime(result_df['date'])
result_df['date'] = pd.to_datetime(result_df['date'], utc=True).dt.date

# Convert "date" column to type str
result_df['date'].astype(str)

# Reset index to date
result_df.set_index("date", inplace=True)

# Store the results as csv file
# result_df.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\sp500_kpis.csv")
