import pandas as pd

table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
# df.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\sp500_constituents.csv")
