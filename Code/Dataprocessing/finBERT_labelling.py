from datasets import load_dataset
from transformers import pipeline

# Read dataset
d = load_dataset("csv", data_files="C:/Users/chris/IdeaProjects/masterProject/Dataset/analyst_ratings_with_price.csv")

# Initialize the pipeline with finBERT
pipe = pipeline(model="ProsusAI/finbert", tokenizer='ProsusAI/finbert')


# Define labelling function which gets mapped to the dataset
def label(record):
    record["finBERT"] = pipe(record['title'])[0]['label']

    return record


# Select a portion of the dataset using slicing
portion = d["train"].select(range(0, 20))

# Map the defined function to the entire dataset
portion = portion.map(label)

# Convert the dataset to a pandas DataFrame
df = portion.to_pandas()

# Save the DataFrame as a CSV file
df.to_csv("path/to/save/dataset.csv", index=False)
