import pandas as pd
import numpy as np
import ressources as r

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


########################################################################################################################
# Concatenate all company info files
########################################################################################################################

# Create empty data frame with required columns
comp_info = pd.DataFrame(columns=["date", "stock", "open", "high", "low", "close", "adj close", "volume"])

for ticker in r.company_tickers:
    # Read company info
    file = r"C:\Users\chris\IdeaProjects\masterProject\Dataset\Company info\\" + ticker + ".csv"
    df2 = pd.read_csv(file)

    # Add company ticker to allow merging on data and ticker
    df2.insert(1, 'stock', ticker)

    # Lower case to allow merging on data and ticker
    df2.rename(str.lower, axis='columns', inplace=True)

    # Calculate return and log return
    df2["return"] = df2["adj close"].pct_change()
    df2["log_return"] = np.log(1 + df2["return"])

    # Concatenate all companies with single company
    comp_info = pd.concat([comp_info, df2])

    # Delete variables
    del file
    del df2


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


########################################################################################################################
# Store dataframe
########################################################################################################################

# df.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\analyst_ratings_with_price.csv")
