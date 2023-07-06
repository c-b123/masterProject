from dataprocessing import StockPriceDataset
import numpy as np
from torch.utils.data import DataLoader
import torch
from model import StockPriceTransformer
import torch.nn as nn
import torch.optim as optim

############## INSTANTIATE DATASET #################
with open('processed_stock_dataset.json') as f:
    dataset = json.load(f)
    dataset = json.loads(dataset)

data = np.array(dataset['open']['data'])[:10, :30]
row_sums = data.sum(axis=1)
data = data / row_sums[:, np.newaxis]

window_size = 5  # number of days to look at
batch_size = 2

dataset = StockPriceDataset(data, window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

################## INSTANTIATE MODEL ################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = window_size * data.shape[0] + data.shape[0]  #
nhead = 8
num_layers = 5
dim_feedforward = 256
num_companies = data.shape[0]
embedding_length = 128

model = StockPriceTransformer(input_size, nhead, num_layers, dim_feedforward, embedding_length).to(device)
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
