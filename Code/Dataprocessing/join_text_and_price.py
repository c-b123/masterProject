import pandas as pd
import stockdata as sd

########################################################################################################################
# Prepare ar_processed.csv for merge with price data
########################################################################################################################

# Read analyst_ratings_processed file
file_name = r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_train.csv"
df = pd.read_csv(file_name)

# Drop rows containing nan
df.dropna(inplace=True)

# Convert from datetime to date format
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], utc=True).dt.date

# Convert "date" column to type str to allow merging
df['date'].astype(str)


########################################################################################################################
# Concatenate stock data into on large file
########################################################################################################################

# Create empty data frame with required columns
comp_info = pd.DataFrame(columns=["date", "stock", "open", "high", "low", "close", "adj close", "volume", "return"])

# Concatenate each stock data file to one large stock file
for ticker in df["stock"].unique().tolist():
    # Fetch stock data from yahoo
    sdata = sd.get_stock_data(ticker, start_date="2022-03-01", end_date="2023-07-25")

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

# df.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_train.csv", index=False)
