import ast
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from Code import resources as r


def convert_date(date_string):
    parsed_date = datetime.strptime(date_string, "%Y%m%dT%H%M%S")
    formatted_date = parsed_date.strftime("%Y-%m-%d")
    return formatted_date


def get_ticker_with_max_relevance(ticker_list):
    tickers = ast.literal_eval(ticker_list)
    max_relevance = max(tickers, key=lambda x: float(x['relevance_score']))
    return max_relevance["ticker"], float(max_relevance["relevance_score"])


# Load dataset
df = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_raw.csv")

# Drop duplicates
df = df.drop_duplicates(subset="summary")

# Rename column
df.rename(columns={'time_published': 'date'}, inplace=True)

# Convert date format
df['date'] = df['date'].apply(lambda x: convert_date(x))

# Assign ticker with max relevance to news
df['stock'], df['relevance'] = zip(*df['ticker_sentiment'].apply(get_ticker_with_max_relevance))

# Drop all stocks which are not of interest
df = df[df['stock'].isin(r.tickers)]

# Only consider news with a relevance score higher than 0.2
df = df.loc[df['relevance'] > 0.2]

# Balance the number of news per stock
grouped = df.groupby('stock')
news_per_stock = 1252
selected_rows = []
for group_name, group_df in grouped:
    if len(group_df) >= 1000:
        selected_rows.append(group_df.sample(n=news_per_stock))
    else:
        selected_rows.append(group_df)
df = pd.concat(selected_rows)

# Print dataset length
print(f"Dataset size: {len(df)}")

# Histogram of stock distribution
hist_sto = df["stock"].value_counts().plot(kind='bar')
hist_sto.set_xticklabels(hist_sto.get_xticklabels(), rotation=0)
plt.xlabel('Stocks')
plt.ylabel('Count')
plt.title('Stock News Distribution')
plt.show()

# Histogram of relevance distribution
# df["relevance"].value_counts().plot(kind='bar')
bin_rel = [0.2, 0.4, 0.6, 0.8, 1]
bin_rel_labels = ['<0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
hist_rel = pd.cut(df['relevance'], bins=bin_rel, labels=bin_rel_labels, right=False).value_counts().sort_index().plot(
    kind='bar')
hist_rel.set_xticklabels(hist_rel.get_xticklabels(), rotation=0)
# Add labels and title to the plot for better understanding
plt.xlabel('Relevance Bins')
plt.ylabel('Count')
plt.title('Relevance Score Distribution')
plt.show()

# Save as csv file
df.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_clean.csv", index=False,
          columns=["title", "date", "summary", "stock", "relevance"])
