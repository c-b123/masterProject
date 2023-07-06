import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
import pandas as pd


class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, d_model, window_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pos_matrix = torch.zeros((seq_length, d_model))

        rec_length = window_size + 1
        for k in range(seq_length):
            pos = k % rec_length
            for j in range(int(d_model / 2)):
                denom = max_len ** (2 * j / d_model)
                pos_matrix[k, 2 * j] = np.sin(pos / denom)
                pos_matrix[k, 2 * j + 1] = np.cos(pos / denom)

        self.register_buffer('pe', pos_matrix)
        cax = plt.matshow(pos_matrix)
        plt.gcf().colorbar(cax)
        plt.show()

    def forward(self, x):
        return x + self.pe


class StockPriceTransformer(nn.Module):
    def __init__(self, input_size, nhead, num_layers, dim_feedforward, embedding_length, window_size):
        super(StockPriceTransformer, self).__init__()

        # 1 x 40 -> where 40 is the sequence length
        # 1 is batch size
        # now we need to add dimension layer

        self.positional_encoding = PositionalEncoding(input_size, embedding_length, window_size)
        self.embedding_length = embedding_length
        self.embedding = nn.Linear(input_size, input_size * embedding_length)
        self.relu = nn.ReLU()

        self.encoder_layer = nn.TransformerEncoderLayer(
            embedding_length,
            nhead,
            dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.linear_1 = nn.Linear(embedding_length, input_size)
        self.linear_2 = nn.Linear(input_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, src):
        batch_size, seq_length = src.size()

        src = self.embedding(src)
        src = self.relu(src)
        src = src.view(batch_size, seq_length, self.embedding_length)

        src = self.positional_encoding(src)

        output = self.encoder(src)
        output = self.linear_1(output)
        output = self.tanh(output)
        output = self.linear_2(output)
        output = self.tanh(output)
        output = output.squeeze()
        return output


class StockPriceDataset(Dataset):
    def __init__(self, data, window_size, mask_prob=0.5, mask_value=-1, sep_token=-0.02):
        self.data = data
        self.window_size = window_size
        self.mask_prob = mask_prob
        self.mask_value = mask_value
        self.sep_token = sep_token

    def __len__(self):
        return self.data.shape[1] - self.window_size + 1

    def __getitem__(self, index):

        start = index
        end = index + self.window_size
        window = self.data[:, start:end]

        masked_window, labels = self.mask_window(window)

        # Add separator tokens to the input
        sep_tokens = torch.tensor([i * self.sep_token for i in range(1, window.shape[0] + 1)]).view(-1, 1)

        masked_window_with_sep = torch.cat((sep_tokens, masked_window), dim=1)
        labels = torch.cat((sep_tokens, labels), dim=1)

        return masked_window_with_sep.flatten(), labels.flatten()

    def mask_window(self, window):
        labels = []
        masked_window = []
        for row in window:
            masked_row = []
            label_row = []
            for val in row:
                if np.random.rand() < self.mask_prob:
                    masked_row.append(self.mask_value)
                    label_row.append(val)
                else:
                    masked_row.append(val)
                    label_row.append(val)
            masked_window.append(masked_row)
            labels.append(label_row)

        return torch.tensor(masked_window).float(), torch.tensor(labels).float()


############## INSTANTIATE DATASET #################
# data = np.random.rand(3, 4)  # Replace with your actual stock price data


with open('JNJ.json') as f:
    dataset = json.load(f)
    # MAKE A FOR LOOP HERE FOR EACH KPI
    df = json.loads(dataset['Open'])
    print(df.keys())
    list_of_dataframes = []
    for key in df:
        temp_df = pd.json_normalize(df[key])
        list_of_dataframes.append(temp_df)

    new_df = pd.concat(list_of_dataframes)
    print(new_df)

# TURN INTO NP ARRAY; PASS ALONG KEYS AS LIST
# MAKE APPROP CHANGES TO DICT 


exit()

data = np.array(dataset['open']['data'])
row_sums = data.sum(axis=1)
data = data / row_sums[:, np.newaxis]
# print(data) # normalize data 


window_size = 5  # number of days to look at
batch_size = 2

dataset = StockPriceDataset(data, window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

################## INSTANTIATE MODEL ################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = window_size * data.shape[0] + data.shape[0]  #

nhead = 8
num_layers = 10
dim_feedforward = 256
num_companies = data.shape[0]
embedding_length = 128

model = StockPriceTransformer(input_size, nhead, num_layers, dim_feedforward, embedding_length, window_size).to(device)
loss_function = nn.MSELoss()  # change loss function to L1
optimizer = optim.Adam(model.parameters(), lr=0.01)

# num_companies
num_epochs = 15

for epoch in range(num_epochs):
    for batch_idx, (masked_data, labels) in enumerate(dataloader):
        masked_data, labels = masked_data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(masked_data)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
