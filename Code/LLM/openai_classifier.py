import time
import openai
import pandas as pd
import keys

# Define api key
openai.api_key = keys.oa_api_key

# Print all available models
models = openai.Model.list()
for model in models["data"]:
    print(model.id)

# Read dataset
d = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_labelled_market.csv", index_col=0)
d = d[:100]


# Function to get sentiment from GPT-3.5 using OpenAI API
def get_sentiment(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are an AI language model trained to analyze and detect the sentiment of news summaries."},
            {'role': 'user',
             'content': f"Analyze the following news summary and determine if the sentiment is: positive, negative or"
                        f"neutral. Return only a single word, either positive, negative or neutral: {text}"}],
        max_tokens=3,
        n=1,
        stop=None,
        temperature=0
    )
    sentiment = response.choices[0].message.content
    time.sleep(21)

    return str.lower(sentiment)


# Apply get_sentiment to dataset
d['gpt-3.5-turbo'] = d['summary'].apply(get_sentiment)

d.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_labelled_market_gpt35.csv")
