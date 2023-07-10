import pandas as pd
import stockdata as sd

########################################################################################################################
# Prepare analyst_ratings_processed.csv for merge with price data
########################################################################################################################

# Read analyst_ratings_processed file
file_name = r"C:\Users\chris\IdeaProjects\masterProject\Dataset\analyst_ratings_processed.csv"
df = pd.read_csv(file_name, usecols=[1, 2, 3])

# Drop rows containing nan
df.dropna(inplace=True)

# Convert from datetime to date format
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], utc=True).dt.date

# Convert "date" column to type str to allow merging
df['date'].astype(str)

# Drop stocks which have less than 360 (1 year) unique dates
df = df.groupby('stock').filter(lambda x: x['date'].nunique() >= 360)


########################################################################################################################
# Concatenate stock data into on large file
########################################################################################################################

# Create empty data frame with required columns
comp_info = pd.DataFrame(columns=["date", "stock", "open", "high", "low", "close", "adj close", "volume", "return",
                                  "log_return"])

# Concatenate each stock data file to one large stock file
for ticker in df["stock"].unique().tolist()[0:9]:
    # Fetch stock data from yahoo
    sdata = sd.get_stock_data(ticker, start_date="2010-01-01", end_date="2020-12-31")

    # Add company ticker to allow merging on data and ticker
    sdata.insert(1, 'stock', ticker)

    # Concatenate all companies with single company
    comp_info = pd.concat([comp_info, sdata])

    # Delete auxiliary dataframe
    del sdata


########################################################################################################################
# Merge company info and text data
########################################################################################################################

# Convert data type to str to allow merging
df = df.astype(str)
comp_info = comp_info.astype(str)

# Merge text data with the corresponding price data
df = pd.merge(df, comp_info, on=["date", "stock"], how='left')

# Drop rows containing no price information
df.replace(["NaN", "nan"], pd.NA, inplace=True)
df.dropna(inplace=True, ignore_index=True)

# Delete auxiliary dataframe
del comp_info


########################################################################################################################
# Store dataframe as csv
########################################################################################################################

# df.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\analyst_ratings_with_price.csv")
