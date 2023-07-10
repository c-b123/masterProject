import torch.nn as nn


class StockPriceTransformer(nn.Module):
    def __init__(self, input_size, nhead, num_layers, dim_feedforward, embedding_length):
        super(StockPriceTransformer, self).__init__()

        # self.positional_encoding = PositionalEncoding(embedding_length)
        self.embedding_length = embedding_length
        self.embedding = nn.Linear(input_size, input_size * embedding_length)
        self.relu = nn.ReLU()

        # Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            embedding_length,
            nhead,
            dim_feedforward,
            batch_first=True
        )
        # Repeating blocks of encoders
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        # Linear layer
        self.linear_1 = nn.Linear(embedding_length, input_size)
        self.linear_2 = nn.Linear(input_size, 1)
        # Instead of softmax
        self.tanh = nn.Tanh()

    def forward(self, src):
        batch_size, seq_length = src.size()

        src = self.embedding(src)
        src = self.relu(src)

        src = src.view(batch_size, seq_length, self.embedding_length)

        # src = self.positional_encoding(src)

        output = self.encoder(src)
        output = self.linear_1(output)
        output = self.tanh(output)
        output = self.linear_2(output)
        output = self.tanh(output)
        output = output.squeeze()
