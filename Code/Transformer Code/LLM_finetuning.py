import pandas as pd
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import *
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from torch import nn, optim
import evaluate

from sklearn.metrics import accuracy_score, f1_score





class Classifier(nn.Module):
    def __init__(self, embedding_length_1, embedding_length_2, num_labels):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embedding_length_1 + embedding_length_2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_labels)
        
    def forward(self, x1, x2):
        # Concatenate the feature vectors along dimension 1
        x = torch.cat((x1, x2), dim=1)
        
        # Pass the concatenated vectors through the MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def collate_fn(batch):

    label_to_id = {"negative": 0, "neutral": 1, "positive": 2}


    input_ids = pad_sequence([Tensor(item['input_ids']) for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([Tensor(item['attention_mask']) for item in batch], batch_first=True, padding_value=0) # DO NOT ATTEND TOKEN
    labels = torch.tensor([label_to_id[item['label']] for item in batch]) # DO NOT ADD TO LOSS TOKEN
    try:
        stock_info = torch.tensor(np.array([item['stock_info'].astype(np.float32).flatten() for item in batch]))
    except Exception as e:
        print(e)
        temp_list = []
        for item in batch:
            print('Here!')
            print(item['stock_info'].shape)

        exit()

    return {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'labels': labels.long(),
        'stock_info': stock_info.long()
    }

class News_Stock_Dataset(Dataset):

    def __init__(self, news_data, stock_data, window_size, tokenizer):



        self.stock_data = self.process_stock_data(stock_data)
        #self.stock_data = stock_data
        self.num_days = int((window_size - 1)/2)
        self.company_similarity_mapping = {
            'MRK':['PFE', 'GSK', 'SNY', 'MRK.DE'],
            'MS':['GS', 'JPM', 'UBS', 'WFC'],
            'MU':['000660.KS', 'INTC', 'WDC', 'NVDA'],
            'NVDA':['AMD', 'MSFT', '2357.TW', 'INTC'],
            'EBAY':['AMZN', 'WMT', 'TGT', 'HM-B.ST'],
            'NFLX':['DIS', 'AAPL', 'MSFT', 'AMZN'],
            'QQQ':['DJIA', 'SPX', 'IXIC', 'RUT'],
            'VZ' : ['T', 'USM', 'LUMN', 'FYBR'],
            'DAL':['LUV', 'KLM', 'JBLU', 'LHA.DE'],
            'JNJ':['NVS', 'ABT', 'PFE', 'GSK'],
            'QCOM':['MDTKF', 'NVDA', 'INTC', 'AMD'],
        }
        self.data = news_data[['date', 'summary', 'stock', 'finBERT']]
        self.tokenizer = tokenizer


    def __len__(self):
         
        return len(self.data)
        
    def __getitem__(self, index):

        
        ticker = self.data['stock'].iloc[index]
        date = self.data['date'].iloc[index]
        label = self.data['finBERT'].iloc[index]

        news_text = self.data['summary'].iloc[index]
        tokenized_text = self.tokenizer(news_text, padding=True, truncation=True)

        stock_info = self.get_stock_data(date, ticker)

        return {"input_ids": tokenized_text['input_ids'], "attention_mask": tokenized_text["attention_mask"], "stock_info": stock_info, "label": label}

    def get_stock_data(self, date, ticker):

        df_dicts = self.stock_data[ticker]
        stock_data_matrix = np.array([])
        for kpi in df_dicts:

            df = df_dicts[kpi]
            df = pd.concat(df.values(), keys=df.keys(), axis=0)
            df.reset_index(inplace=True)
            df = df.T
            df.columns = df.iloc[0]
            df = df.iloc[2:]

            df = df.sort_values('Date')
            loc = df.index.get_loc(df[df['Date'] == date].index[0])
            df = df.drop('Date', axis=1)

            slice_df = df.iloc[max(0, loc - self.num_days):min(loc+ (self.num_days + 1), len(df))]

            numpy_array = slice_df.to_numpy()
            try:
                stock_data_matrix = np.concatenate((stock_data_matrix, numpy_array.T), axis=1)
            except Exception as e:
                # print(e)
                stock_data_matrix = numpy_array.T

        return stock_data_matrix
    

    def process_stock_data(self, stock_data):
        ####### Normalize the dataframes 
        for comp_  in stock_data:
            dataset = stock_data[comp_]
            for kpi in dataset.keys():
                
                try:
                    df = json.loads(dataset[kpi])
                except:
                    continue
                for second_key in df.keys():
                    temp_df = pd.json_normalize(df[second_key])
                    if second_key.strip() == "Date":
                        pass
                    else:
                        
                        min_ = temp_df.min(axis=1)[0]
                        max_ = temp_df.max(axis=1)[0]
                        temp_df = (temp_df - min_)/(max_ - min_)
                    df[second_key] = temp_df
                
                dataset[kpi] = df
            
            stock_data[comp_] = dataset
        
        return stock_data




def filter_news_df(news_df, stock_df):

    to_remove = []
    for i in range(len(news_df)):
        stock_ticker = news_df['stock'].iloc[i]
        date = news_df['date'].iloc[i]
        try:
            comp_df =   stock_df[stock_ticker]
        except:
            to_remove.append(i)
            continue
        
       
        stock_dates = list(json.loads(comp_df["Open"])['Date'].values())
        stock_dates.sort()
        try:
            idx = stock_dates.index(date)
        except:
            to_remove.append(i)
            continue
        if (idx - 2) < 0 or (idx + 2) >= len(stock_dates):
            to_remove.append(i)
    
        # assuming df is your DataFrame and indices_to_drop is your list of indices
    news_df = news_df.drop(to_remove)
    return news_df

def compute_metrics(logits, labels):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")

    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='weighted')["f1"]

    return {"accuracy": [accuracy], "f1": [f1]}




########### DATASET CREATION



dataset = {}
dir_file = './Data/Stock Data/'
for filename in os.listdir(dir_file):
    if filename.endswith('.json'):
        with open(os.path.join(dir_file, filename)) as f:
            dataset[filename.replace(".json", "")] = json.load(f)

window_size = 5
batch_size = 2
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
news_df = pd.read_csv("./Data/av_train.csv")
news_df = filter_news_df(news_df, dataset)

# comps = ["NFLX"] #"MS", "JNJ"
# news_df = news_df[news_df['stock'].isin(comps)] # , MS, JNJ



# print(json.loads(dataset['NVDA']['Open'])['Date'])

train_news, val_news = train_test_split(news_df, test_size=0.1, random_state = 518)


train_news_stock_dataset = News_Stock_Dataset(train_news, dataset, window_size, tokenizer)
val_news_stock_dataset = News_Stock_Dataset(val_news, dataset, window_size, tokenizer)
train_dataloader = DataLoader(train_news_stock_dataset, batch_size=1, collate_fn =collate_fn)
eval_dataloader = DataLoader(val_news_stock_dataset, batch_size=1, collate_fn = collate_fn)

####### MODELS
nhead = 4
num_layers = 2
dim_feedforward = 128
num_companies = 5
embedding_length = 4
num_kpis = 6
input_size = (window_size * num_kpis )  * num_companies
stock_model = StockPriceTransformer(input_size, nhead, num_layers, dim_feedforward, embedding_length, window_size, get_embeddings = True)
model_path = "experiments/20230801185820/model_epoch_18.pth"
model_state = torch.load(model_path)
stock_model.load_state_dict(model_state)

lang_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Create the classifier
embedding_length_1 = 768  # Replace with your value
embedding_length_2 = 600  # Replace with your value
num_labels = 3
classifier_model = Classifier(embedding_length_1, embedding_length_2, num_labels)
optimizer = optim.AdamW(classifier_model.parameters(), lr=0.0005)
loss_function =  nn.CrossEntropyLoss()


# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO)

# Create a directory for this experiment based on the current timestamp
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
experiment_dir = f'./experiments_pipeline/{timestamp}'
os.makedirs(experiment_dir, exist_ok=True)

# Set up TensorBoard
writer = SummaryWriter(log_dir=experiment_dir)



"""

num_epochs = 15
for epoch in range(num_epochs):


    classifier_model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_dataloader):


        

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        stock_info = batch['stock_info']
        stock_info = stock_info.to(torch.float32)
        # print(stock_info.size())




        with torch.no_grad():
            outputs = lang_model(input_ids = input_ids, attention_mask = attention_mask)
            stock_embeddings = stock_model(stock_info)


        ############ LANGUAGE Embeddings
        last_hidden_state = outputs.last_hidden_state
        cls_embeddings = last_hidden_state[:, 0, :]
        stock_embeddings = stock_embeddings.view(1, 600)

        output = classifier_model(cls_embeddings, stock_embeddings)
        optimizer.zero_grad()
        loss = loss_function(output, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Training Loss Step', loss.item(), batch_idx)
    

    
    training_loss = running_loss / len(train_dataloader)
    writer.add_scalar('Training Loss', training_loss, epoch)

    classifier_model.eval()
    with torch.no_grad():

        running_loss = 0.0

        for batch_idx, val_batch in enumerate(eval_dataloader):

            input_ids = val_batch['input_ids'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            labels = val_batch['labels'].to(device)
            stock_info = val_batch['stock_info']
            stock_info = stock_info.to(torch.float32)

            outputs = lang_model(input_ids = input_ids, attention_mask = attention_mask)
            try:
                stock_embeddings = stock_model(stock_info)
            except:
                print(stock_info.shape)

            last_hidden_state = outputs.last_hidden_state
            cls_embeddings = last_hidden_state[:, 0, :]

            try:
                stock_embeddings = stock_embeddings.view(1, 600)
            except:
                print(stock_embeddings.size())
                continue
            output = classifier_model(cls_embeddings, stock_embeddings)
            loss = loss_function(output, labels)
            running_loss += loss.item()


        validation_loss = running_loss / len(eval_dataloader)
        writer.add_scalar('Validation Loss', validation_loss, epoch)


        print(f"Epoch: {epoch+1}, Training Loss: {training_loss}, Validation Loss: {validation_loss}")
        torch.save(classifier_model.state_dict(), os.path.join(experiment_dir, f'model_epoch_{epoch+1}.pth'))

exit()
"""
###### Load test set; create Loop evaluate etc. 
news_df = pd.read_csv("./Data/av_test.csv")
dataset = {}
dir_file = './Data/Stock Data/'
for filename in os.listdir(dir_file):
    if filename.endswith('.json'):
        with open(os.path.join(dir_file, filename)) as f:
            dataset[filename.replace(".json", "")] = json.load(f)




news_df = filter_news_df(news_df, dataset)
test_news_stock_dataset = News_Stock_Dataset(news_df, dataset, window_size, tokenizer)
test_dataloader = DataLoader(test_news_stock_dataset, batch_size=1, collate_fn =collate_fn)
print(len(news_df))



embedding_length_1 = 768  # Replace with your value
embedding_length_2 = 600  # Replace with your value
num_labels = 3
classifier_model = Classifier(embedding_length_1, embedding_length_2, num_labels)
loss_function =  nn.CrossEntropyLoss()
model_path = "experiments_pipeline/20230801192326/model_epoch_6.pth"
model_state = torch.load(model_path)
classifier_model.load_state_dict(model_state)

all_preds = []
all_labels = []
with torch.no_grad():

        running_loss = 0.0
        for batch_idx, val_batch in enumerate(test_dataloader):
            print(batch_idx)

            input_ids = val_batch['input_ids'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            labels = val_batch['labels'].to(device)
            stock_info = val_batch['stock_info']
            stock_info = stock_info.to(torch.float32)

            outputs = lang_model(input_ids = input_ids, attention_mask = attention_mask)
            try:
                stock_embeddings = stock_model(stock_info)
            except:
                print(stock_info.shape)

            last_hidden_state = outputs.last_hidden_state
            cls_embeddings = last_hidden_state[:, 0, :]

            try:
                stock_embeddings = stock_embeddings.view(1, 600)
            except:
                print(stock_embeddings.size())
                continue
            output = classifier_model(cls_embeddings, stock_embeddings)
            preds = np.argmax(output, axis=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
