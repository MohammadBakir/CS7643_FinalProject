import torch
import torch.nn as nn
import torch.nn.functional as f


def attention(query, key, value):
    temp = query.bmm(key.transpose(1, 2))
    softmax = f.softmax(temp, dim=-1)
    attention = softmax.bmm(value)
    return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, key_dim, value_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, key_dim)
        self.key = nn.Linear(input_dim, key_dim)
        self.value = nn.Linear(input_dim, value_dim)
        self.num_heads = num_heads
        self.linear = nn.Linear(num_heads * value_dim, input_dim)

    def forward(self, query, key, value):
        multiheads_out = [
            attention(self.query(query), self.key(key), self.value(value)) for _ in
            range(self.num_heads)
        ]
        out = self.linear(torch.cat(multiheads_out, dim=-1))
        return out


# todo should we instead implement learnable embedding
def positioning_encoding(device, seq_length, model_dim):
    position = torch.arange(seq_length, dtype=torch.int).reshape(1, -1, 1).to(device)
    frequencies = 1e-4 ** (2 * (torch.div(torch.arange(model_dim, dtype=torch.int), 2)) / model_dim).reshape(1, 1,
                                                                                                             -1).to(
        device)
    pos_enc = position * frequencies
    pos_enc[:, ::2] = torch.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = torch.sin(pos_enc[:, 1::2])
    return pos_enc


def forward(input_dim=512, forward_dim=2048):
    forward_layer = nn.Sequential(
        nn.Linear(input_dim, forward_dim),
        nn.ReLU(),
        nn.Linear(forward_dim, input_dim)
    )
    return forward_layer


class ResidualConnection(nn.Module):
    def __init__(self, layer, dimension, dropout=0.2):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *X):
        return self.norm(X[-1] + self.dropout(self.layer(*X)))


class Encoder(nn.Module):
    def __init__(self, device, n_layers=6, model_dim=512, num_heads=8, forward_dim=2048, dropout=0.2):
        super().__init__()
        self.device = device
        self.n_layers = n_layers
        key_dim = value_dim = model_dim // num_heads
        self.multihead_attention = ResidualConnection(
            MultiHeadAttention(num_heads, model_dim, key_dim, value_dim),
            dimension=model_dim,
            dropout=dropout
        )
        self.feed_forward = ResidualConnection(
            forward(model_dim, forward_dim),
            dimension=model_dim,
            dropout=dropout
        )

    def forward(self, X):
        seq_length, dimension = X.size(1), X.size(2)
        out = X
        out += positioning_encoding(self.device, seq_length, dimension)
        for _ in range(self.n_layers):
            att_out = self.multihead_attention(out, out, out)
            out = self.feed_forward(att_out)
        return out


class TransformerModel(nn.Module):
    def __init__(self, device, n_layers=6, model_dim=512, output_dim=512,
                 num_heads=6, forward_dim=2048, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(device, n_layers, model_dim, num_heads, forward_dim, dropout)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(model_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        enc_out = self.encoder(X)
        out = self.relu(self.linear(enc_out[:, -1, :]))
        return out


class TransformerModelImpl(nn.Module):
    def __init__(self, params):
        super(TransformerModelImpl, self).__init__()
        self.transf = TransformerModel(device=params.DEVICE,
                                       n_layers=params.TMI_N_LAYERS,
                                       num_heads=params.TMI_NUM_HEADS,
                                       model_dim=params.FEATURES,
                                       forward_dim=params.TMI_FORWARD_DIM,
                                       output_dim=params.TMI_OUTPUT_DIM,
                                       dropout=params.DROPOUT)
        self.linear = nn.Linear(params.TMI_OUTPUT_DIM, params.OUTPUT_DIM)

    def forward(self, x):
        transf_out = self.transf(x)
        out = self.linear(transf_out)
        return out


class TransformerModelImpl2(nn.Module):
    def __init__(self, params):
        super(TransformerModelImpl2, self).__init__()
        self.device = params.device
        self.position_emb = nn.Embedding(params.seq_len, params.d_model)
        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, dropout=params.dropout, d_model=params.d_model,
                                                   nhead=params.nhead,
                                                   dim_feedforward=params.dim_feedforward)  # could help to use: norm_first=True
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=params.num_layers)
        self.fc_layer = nn.Linear(params.d_model * params.seq_len, params.d_output)

    def forward(self, X):
        seq_length, dimension = X.size(1), X.size(2)
        out = X
        lookup = torch.arange(seq_length, dtype=torch.int).to(self.device)
        out += self.position_emb(lookup)
        out = self.transformer_encoder(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc_layer(out)
        return out.reshape(X.shape[0], -1)
