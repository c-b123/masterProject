from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import pipeline
from huggingface_hub import login
import keys

login(token=keys.hf_api_key)

########################################################################################################################
# This script needs GPU acceleration e.g. via Google Colab
########################################################################################################################


# Create a pipeline with Llama 2
pipe = pipeline("text-generation", model="meta-llama/Llama-2-70b-chat-hf")


# Define dataset structure
class NewsDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        res = f"Analyze the following news summary and determine if the sentiment is: positive, negative or neutral.\n" \
              f"{self.original_list[i]}\n" \
              f"Return only a single word, either positive, negative or neutral."
        return res


# Load dataset
d = load_dataset("csv", data_files="/content/drive/MyDrive/masterProject/av_labelled_market.csv")
d = d["train"][:3]

# Instantiate dataset
dataset = NewsDataset(d["summary"])

# # Find optimal batch size on subset of data
# for batch_size in [1, 8, 64, 256]:
#     print("-" * 30)
#     print(f"Streaming batch_size={batch_size}")
#     for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
#         pass


# Label text data
results = []
for out in tqdm(pipe(dataset, batch_size=8, max_length=3, num_return_sequences=1), total=len(dataset)):
    results.append(out["label"])

# Add results to the original dataset
d = d.add_column("Llama2", results)

# Save the DataFrame as a CSV file
d.to_csv("/content/drive/MyDrive/masterProject/av_labelled_market_Llama2.csv", index=False)
