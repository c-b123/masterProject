from transformers import pipeline
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm

########################################################################################################################
# This script needs GPU acceleration e.g. via Google Colab
########################################################################################################################


# Define dataset structure
class ArDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


# Load dataset
d = load_dataset("csv", data_files="/content/drive/MyDrive/masterProject/analyst_ratings_with_price.csv")
d = d["train"]

# Instantiate dataset
dataset = ArDataset(d["title"])

# Find optimal batch size on subset of data
# for batch_size in [1, 8, 64, 256]:
#     print("-" * 30)
#     print(f"Streaming batch_size={batch_size}")
#     for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
#         pass

# Create a pipeline using the desired model and tokenizer using a GPU (device=0)
pipe = pipeline(model="ProsusAI/finbert", tokenizer='ProsusAI/finbert', device=0)

# Label text data
results = []
for out in tqdm(pipe(dataset, batch_size=8), total=len(dataset)):
    results.append(out["label"])

# Add results to the original dataset
d = d.add_column("finBERT", results)

# Save the DataFrame as a CSV file
# d.to_csv("/content/drive/MyDrive/masterProject/analyst_ratings_labelled.csv", index=False)
