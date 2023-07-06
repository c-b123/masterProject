import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


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
