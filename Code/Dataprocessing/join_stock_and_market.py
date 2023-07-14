import pandas as pd

# Read ar_labelled, containing stock data and finBERT labels
ar_labelled = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\ar_labelled.csv", index_col=0)

# Read sp500_kpis, containing market data
sp500_kpis = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\sp500_kpis.csv")

# Merge both datasets on date
ar_labelled_market = pd.merge(ar_labelled, sp500_kpis, how="left", on="date")

# Rename the column names for better understanding
ar_labelled_market.rename(columns={'mean': "sp_mean", 'variance': "sp_var", '10th_percentile': "sp_10_pct",
                                   '25th_percentile': "sp_25_pct", '50th_percentile': "sp_50_pct",
                                   '75th_percentile': "sp_75_pct", '90th_percentile': "sp_90_pct"},
                          errors="raise", inplace=True)

# Export merged dataset as csv
ar_labelled_market.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\ar_labelled_market.csv")
