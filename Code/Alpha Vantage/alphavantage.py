import json
import requests

api_key = "5224NQCCUYJH71Z3"


def get_av_news_data(tickers: list, topics: list, time_from: str, time_to: str, sort: bool, limit: int, apikey: str):

    return


# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&limit=25&tickers=AAPL&apikey=5224NQCCUYJH71Z3'
r = requests.get(url)
data = r.json()

print(data)
filtered_records = [record for record in data["feed"] if
                    len(record['ticker_sentiment']) == 1 and
                    float(record["ticker_sentiment"][0]["relevance_score"]) > 0.5]

# Print the filtered records
for record in filtered_records:
    print(record)
with open(r'C:\Users\chris\IdeaProjects\masterProject\Dataset\av_data.json', 'w') as file:
    json.dump(data, file, indent=3)
print(type(r))
print(type(data))


