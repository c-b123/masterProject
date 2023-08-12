import logging
import os
import torch
import json
from torch import nn, optim
from torch.utils.data import DataLoader
from dataprocessing import StockPriceDataset
from model import StockPriceTransformer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

window_size = 5
batch_size = 2

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO)

# Create a directory for this experiment based on the current timestamp
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
experiment_dir = f'./experiments/{timestamp}'
os.makedirs(experiment_dir, exist_ok=True)

# Set up TensorBoard
writer = SummaryWriter(log_dir=experiment_dir)

# Load the training dataset
dataset = []
dir_file = './train_datasets'
for filename in os.listdir(dir_file):
    if filename.endswith('.json'):
        with open(os.path.join(dir_file, filename)) as f:
            dataset.append(json.load(f))

train_dataset = StockPriceDataset(dataset, window_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

# Load the evaluation dataset
dir_file = './eval_datasets'
dataset = []
for filename in os.listdir(dir_file):
    if filename.endswith('.json'):
        with open(os.path.join(dir_file, filename)) as f:
            dataset.append(json.load(f))

eval_dataset = StockPriceDataset(dataset, window_size)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

# # Load the testing dataset
# dir_file = './test_datasets'
# dataset = []
# for filename in os.listdir(dir_file):
#     if filename.endswith('.json'):
#         with open(os.path.join(dir_file, filename)) as f:
#             dataset.append(json.load(f))

# test_dataset = StockPriceDataset(dataset, window_size)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

######## Model Hyperparameters
nhead = 4
num_layers = 2
dim_feedforward = 128
num_companies = 5
embedding_length = 4
num_kpis = 6
input_size = (window_size * num_kpis )  * num_companies


################## INSTANTIATE MODEL ################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StockPriceTransformer(input_size, nhead, num_layers, dim_feedforward, embedding_length, window_size).to(device)
loss_function = nn.L1Loss() # change loss function to L1
optimizer = optim.AdamW(model.parameters(), lr=0.0005)


# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for batch_idx, (masked_data, labels) in enumerate(train_dataloader):
        masked_data, labels = masked_data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(masked_data)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Training Loss Step', loss.item(), batch_idx)

        running_loss += loss.item()

    training_loss = running_loss / len(train_dataloader)
    writer.add_scalar('Training Loss', training_loss, epoch)

    # Validation
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for batch_idx, (masked_data, labels) in enumerate(eval_dataloader):
            masked_data, labels = masked_data.to(device), labels.to(device)
            output = model(masked_data)
            loss = loss_function(output, labels)
            running_loss += loss.item()

        validation_loss = running_loss / len(eval_dataloader)
        writer.add_scalar('Validation Loss', validation_loss, epoch)

    print(f"Epoch: {epoch+1}, Training Loss: {training_loss}, Validation Loss: {validation_loss}")

    # Save the model parameters after each epoch
    torch.save(model.state_dict(), os.path.join(experiment_dir, f'model_epoch_{epoch+1}.pth'))

writer.close()