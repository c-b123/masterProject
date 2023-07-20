import pandas as pd

# Load dataset
d = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_data.csv", index_col=0)

# Do not consider cryptocurrencies or forex trading
d = d[~d['stock'].str.startswith('CRYPTO')]
d = d[~d['stock'].str.startswith('FOREX')]

# Only consider news with a relevance score higher than 0.33
d = d.loc[d['relevance'] > 0.33]
print(len(d))

# Drop stocks with less than 100 news articles
value_counts = d['stock'].value_counts()
values_to_drop = value_counts[value_counts < 100].index
d = d[~d['stock'].isin(values_to_drop)]

# Select 100 random news articles from each stock
grouped = d.groupby('stock')
news_per_stock = 100
selected_rows = []
for group_name, group_df in grouped:
    if len(group_df) >= news_per_stock:
        selected_rows.append(group_df.sample(n=news_per_stock))
d = pd.concat(selected_rows)

# Reset index
d = d.reset_index(drop=True)

abc = d["stock"].value_counts()

# Save as csv file
d.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_data_processed.csv")


