import pandas as pd

file_name = r"C:\Users\chris\IdeaProjects\masterProject\Dataset\analyst_ratings_processed.csv"
df = pd.read_csv(file_name, index_col=0)
df = df.groupby('stock').filter(lambda x: len(x) >= 500)
cnt = df.stock.value_counts()
file = r"C:\Users\chris\IdeaProjects\masterProject\Dataset\sp500.csv"
sp500 = pd.read_csv(file, index_col=0)
# Convert from datetime to date format
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], utc=True).dt.date

# Convert "date" column to type str to allow merging
df['date'].astype(str)
cnt_date = df.groupby("stock")["date"].nunique()
print(df.groupby("stock")["date"].nunique())
