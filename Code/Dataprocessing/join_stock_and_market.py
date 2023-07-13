import pandas as pd

analyst_ratings_labelled = pd.read_csv(
    r"/Dataset/ar_labelled.csv", index_col=0)
sp500_kpis = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\sp500_kpis.csv")

analyst_ratings_labelled = pd.merge(analyst_ratings_labelled, sp500_kpis, how="left", on="date")

analyst_ratings_labelled.rename(columns={'mean': "sp_mean", 'variance': "sp_var", '10th_percentile': "sp_10_pct",
                                         '25th_percentile': "sp_25_pct", '50th_percentile': "sp_50_pct",
                                         '75th_percentile': "sp_75_pct", '90th_percentile': "sp_90_pct"},
                                errors="raise", inplace=True)

analyst_ratings_labelled.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\ar_labelled_market.csv")
