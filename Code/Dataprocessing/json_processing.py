import json
import csv
import codecs
from datetime import datetime


def get_max_relevance_score(record):
    max_score = 0
    max_ticker = ""
    for ticker_sentiment in record["ticker_sentiment"]:
        relevance_score = float(ticker_sentiment["relevance_score"])
        ticker = ticker_sentiment["ticker"]
        if relevance_score > max_score:
            max_score = relevance_score
            max_ticker = ticker
    return max_ticker, max_score


def convert_to_csv(file_path):
    with open(file_path) as file:
        data = json.load(file)

    result = []
    for record in data:
        date = datetime.strptime(record["time_published"], "%Y%m%dT%H%M%S").strftime("%Y-%m-%d")
        ticker, relevance = get_max_relevance_score(record)
        result.append({'title': record["title"], 'summary': record["summary"], 'date': date, 'stock': ticker,
                       'relevance': relevance})

    fieldnames = ['title', 'summary', 'date', 'stock', 'relevance']
    with codecs.open(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_data.csv", 'w', 'utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the headers to the CSV file
        writer.writeheader()

        # Write each record as a row in the CSV file
        writer.writerows(result)

    return result


# d = convert_to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_data.json")
