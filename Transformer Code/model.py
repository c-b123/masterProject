import torch.nn as nn
import torch
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, d_model, window_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pos_matrix = torch.zeros((seq_length, d_model))

        rec_length = window_size + 1
        for k in range(seq_length):
            pos = k % rec_length
            for j in range(int(d_model/2)):
                denom = max_len **(2 * j/d_model)
                pos_matrix[k, 2*j] = np.sin(pos/denom)
                pos_matrix[k, 2*j + 1] = np.cos(pos/denom)
        
        self.register_buffer('pe', pos_matrix)

    def forward(self, x):
        return x + self.pe






class StockPriceTransformer(nn.Module):
    def __init__(self, input_size, nhead, num_layers, dim_feedforward, embedding_length, window_size, get_embeddings = False):
        super(StockPriceTransformer, self).__init__()


        self.get_embedding = get_embeddings
        self.positional_encoding = PositionalEncoding(input_size, embedding_length, window_size)
        self.embedding_length = embedding_length
        self.embedding = nn.Linear(input_size, input_size * embedding_length)
        self.relu = nn.ReLU()

        self.encoder_layer = nn.TransformerEncoderLayer(
            embedding_length,
            nhead,
            dim_feedforward,
            batch_first = True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.linear_1 = nn.Linear(embedding_length, input_size)
        self.linear_2 = nn.Linear(input_size, 1)
        self.tanh = nn.Tanh()

        


    def forward(self, src):

        # print(src.size())

        batch_size, seq_length = src.size()



        src = self.embedding(src)
        src = self.relu(src)
        src = src.view(batch_size, seq_length, self.embedding_length)
        src = self.positional_encoding(src)

        output = self.encoder(src)


        if self.get_embedding:
            return output
        
        output = self.linear_1(output)
        output = self.tanh(output)
        output = self.linear_2(output)
        output = self.tanh(output)
        output = output.squeeze()
        return output

