import time
import openai
import pandas as pd
import keys

# Define api key
openai.api_key = keys.oa_api_key

# # Print all available models
# models = openai.Model.list()
# for model in models["data"]:
#     print(model.id)

# Read dataset
d = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\financial_phrasebank_allagree.csv")
d = d.sample(n=100, random_state=42, ignore_index=True)


# Function to get sentiment from GPT-3.5 using OpenAI API
def get_sentiment(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are an AI language model trained to analyze and detect the sentiment of"
                        "financial news summaries. You consider the provided financial news summaries from the"
                        "view-point of an investor only."},
            {'role': 'user',
             'content': f"Analyze the following financial news summary and determine if it has a positive, negative, or"
                        f"neutral influence on the stock price. Return only a single word, either positive, negative or"
                        f"neutral: {text}"}],
        max_tokens=3,
        n=1,
        stop=None,
        temperature=0
    )
    sentiment = response.choices[0].message.content
    time.sleep(21)

    return str.lower(sentiment)


# Apply get_sentiment to dataset
d['gpt-3.5-turbo'] = d['sentence'].apply(get_sentiment)

d.to_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\financial_phrasebank_allagree_gpt35.csv")
