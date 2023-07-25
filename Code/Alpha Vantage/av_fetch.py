import time
import pandas as pd
import requests
import keys
from datetime import datetime, timedelta
from Code import resources as r


def get_stock_news_date(tickers: list, time_from: str, time_to="", sort="EARLIEST", limit=50):
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


def get_consecutive_stock_news(stock: str, start_date: str):
    fetch_lst = []
    date = start_date
    for i in range(0, 16):
        if datetime.strptime(date, "%Y%m%dT%H%M%S") < datetime.strptime("20230715T000000", "%Y%m%dT%H%M%S"):
            fetch, date = get_stock_news_date(tickers=[stock], time_from=date, limit=1000)
            fetch_lst.extend(fetch)
            time.sleep(1)
        else:
            break
    return fetch_lst


def get_news(start_date: str):
    result = []
    for stock in r.tickers:
        try:
            fetch = get_consecutive_stock_news(stock, start_date)
            result.extend(fetch)
        except Exception as e:
            print("Fetch failed for stock:" + stock)
            print(f"Error details: {e}")
    return result


fetch_all = get_news("20220301T000000")
df = pd.DataFrame(fetch_all)
# df.to_csv(r'C:\Users\chris\IdeaProjects\masterProject\Dataset\av_raw.csv', index=False)
