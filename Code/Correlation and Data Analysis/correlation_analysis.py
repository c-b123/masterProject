import pandas as pd

file_name = r"C:\Users\chris\IdeaProjects\masterProject\Dataset\analyst_ratings_with_price.csv"
df = pd.read_csv(file_name, index_col=0)
