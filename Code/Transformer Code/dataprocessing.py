import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json



class StockPriceDataset(Dataset):
    def __init__(self, data, window_size, mask_prob=0.3, mask_value=-1, sep_token=-0.1):



        
        self.window_size = window_size
        self.mask_prob = mask_prob
        self.mask_value = mask_value
        self.sep_token = sep_token
        self.data = self.process_data(data)
        print("self.list_of_lens: ", self.list_of_lens)
        

    def __len__(self):
         
        return self.list_of_lens[-1] - self.window_size + 1
        
    def __getitem__(self, index):


        # print('index: ', index)
       
        for idx, i in enumerate(self.list_of_lens):
            # print('i: ', i)


            if index <= i:
                data = self.data[idx]
            
                if idx > 0:
                    index = index - self.list_of_lens[idx - 1]# + 1
                
                break

        start = index
        end = index + self.window_size
        for key in data:
            temp_window = data[key][:, start:end]
            try:
                window = np.concatenate((window, temp_window), axis = 1)
            except Exception as e:
                window = temp_window.copy()

        masked_window, labels = self.mask_window(window)
        _, seq_length = masked_window.size()
        if seq_length != 30:
            print(masked_window.size())
            
        # Add separator tokens to the input
        # sep_tokens = torch.tensor([i * self.sep_token for i in range(1, window.shape[0] + 1)]).view(-1, 1)


        # masked_window = torch.cat((sep_tokens, masked_window), dim=1)
        # labels = torch.cat((sep_tokens, labels), dim=1)
        return masked_window.flatten(), labels.flatten()

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
    

    def process_data(self, list_of_datasets):

        list_to_return = []
        list_of_lens = []

        for dataset in list_of_datasets:
            data_dict = {}
            for key in dataset.keys():
                try:
                    df = json.loads(dataset[key])
                except:
                    df = dataset[key]
            
                
                list_of_dataframes = []
                for second_key in df.keys():
                    if second_key.strip() == "Date":
                        continue
                    temp_df = pd.json_normalize(df[second_key])
                    min_ = temp_df.min(axis=1)[0]
                    max_ = temp_df.max(axis=1)[0]
                    temp_df= (temp_df - min_)/(max_ - min_)
                    list_of_dataframes.append(temp_df)
                
                new_df = pd.concat(list_of_dataframes)
                new_df = new_df.to_numpy()
                data_dict[key] = new_df

                _, cols =new_df.shape

            try:
                list_of_lens.append(cols + list_of_lens[-1] - self.window_size)
            except:
                list_of_lens.append(cols - self.window_size)
            
            list_to_return.append(data_dict)
                
        self.list_of_lens = list_of_lens
        return list_to_return
