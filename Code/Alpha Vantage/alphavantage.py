import json
import time
from datetime import datetime, timedelta
import requests


def get_av_news_data(tickers: list, time_from: str, time_to="", sort="EARLIEST", limit=50):
    date_from = datetime.strptime(time_from, "%Y%m%dT%H%M%S")
    nextday = date_from + timedelta(days=1)
    date_from = datetime.strftime(date_from, "%Y%m%dT%H%M")
    if time_to:
        date_to = datetime.strptime(time_to, "%Y%m%dT%H%M")
        date_to = '&time_to=' + datetime.strftime(date_to, "%Y%m%dT%H%M")
    else:
        date_to = ""

    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT' \
          + '&tickers=' + ",".join(tickers) \
          + '&time_from=' + date_from \
          + date_to \
          + '&sort=' + sort \
          + '&limit=' + str(limit) \
          + '&apikey=5224NQCCUYJH71Z3'
    print(url)
    r = requests.get(url)
    if "No articles found." in r.text or "\"items\": \"0\"" in r.text:
        data = {}
        last_date = datetime.strftime(nextday, "%Y%m%dT%H%M")
        print(last_date)
    else:
        data = r.json()
        data = data["feed"]
        last_date = data[-1]["time_published"]
        print(last_date)

    return data, last_date


fetch_all = []
date = "20230301T000000"
for i in range(0, 3):
    fetch, date = get_av_news_data(tickers=["WMT"], time_from=date)
    fetch_all.extend(fetch)
    time.sleep(5)


with open(r'C:\Users\chris\IdeaProjects\masterProject\Dataset\av_data.json', 'w') as file:
    json.dump(fetch_all, file, indent=3)

# # filtered_records = [record for record in data["feed"] if float(record["ticker_sentiment"][0]["relevance_score"]) > 0.5]
# # len(record['ticker_sentiment']) == 1 and
# def filter_data_by_relevance_score(data, ticker):
#     filtered_data = []
#     for record in data["feed"]:
#         max_score = 0
#         max_ticker = ""
#         for ticker_sentiment in record["ticker_sentiment"]:
#             relevance_score = float(ticker_sentiment["relevance_score"])
#             ticker = ticker_sentiment["ticker"]
#             if relevance_score > max_score:
#                 max_score = relevance_score
#                 max_ticker += ticker
#         if max_ticker == ticker:
#             filtered_data.append(record)
#     return filtered_data
#
#
# filtered_records = filter_data_by_relevance_score(data, "MS")
# # Print the filtered records
# cnt = 0
# for record in filtered_records:
#     cnt += 1
# print(cnt)
