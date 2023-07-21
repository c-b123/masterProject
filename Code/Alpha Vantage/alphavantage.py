import json
import time
import requests
from datetime import datetime, timedelta
import keys
from Code.Dataprocessing import resources as r


def get_av_news_data(tickers: list, time_from: str, time_to="", sort="EARLIEST", limit=50):
    date_from = datetime.strptime(time_from, "%Y%m%dT%H%M%S")
    nextday = date_from + timedelta(days=30)
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
          + '&apikey=' + keys.av_api_key
    print(url)
    r = requests.get(url)
    if "No articles found." in r.text or "\"items\": \"0\"" in r.text:
        data = {}
        last_date = datetime.strftime(nextday, "%Y%m%dT%H%M")
        print(tickers[0] + ": No articles found. Last date: " + last_date)
    else:
        data = r.json()
        data = data["feed"]
        last_date = data[-1]["time_published"]
        print(last_date)

    return data, last_date


def get_news_for_stock(stock: str, start_date: str):
    fetch_lst = []
    date = start_date
    for i in range(0, 16):
        if datetime.strptime(date, "%Y%m%dT%H%M%S") < datetime.strptime("20230501T000000", "%Y%m%dT%H%M%S"):
            fetch, date = get_av_news_data(tickers=[stock], time_from=date, limit=1000)
            fetch_lst.extend(fetch)
            time.sleep(1)
        else:
            break
    return fetch_lst


def get_news(start_date: str):
    result = []
    for stock in r.sp500:
        try:
            fetch = get_news_for_stock(stock, start_date)
            result.extend(fetch)
        except Exception as e:
            print("Fetch failed for stock:" + stock)
            print(f"Error details: {e}")
    return result


fetch_all = get_news("20230101T000000")

with open(r'C:\Users\chris\IdeaProjects\masterProject\Dataset\av_data.json', 'w') as file:
    json.dump(fetch_all, file, indent=3)
