import torch
import torch.nn as nn
import torch.nn.functional as f

#todo add citation for transformer

def attention(query, key, value):
    '''
    Computes the local fields and the attention of the inputs as described in Vaswani et. al.
    and then scale it for a total sum of 1

    INPUT: query, key, value - input data of size (batch_size, seq_length, num_features)
    '''
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    attention = softmax.bmm(value)
    return attention
    
class MultiHeadAttention(nn.Module):
    '''
    Computes the multihead head consisting of a feedforward layer for each input value
    where the attention for all of these are computed for each head and then concatenated and projected
    as described in Vaswani et. al.

    INPUT: dimensions of the three matrices (where the key and query matrix has the same dimensions) and the nr of heads
    OUTPUT: the projected output of the multihead attention
    '''

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
        
#todo should we instead implement learnable embedding
def positioning_encoding(device, seq_length, model_dim):
    '''
    Computes the positional encoding for the current state of the elements in the input sequence as
    there is no recurrence or convolution. Using the same encoding with sinusosoidal functions as in Vaswani et. al.
    as the motivations of linearly dependency of the relative positions and the ability to extrapolate to sequence lengths
    longer than encountered in training holds strong.
    Code copied from Frank Odom

    INPUT: length of the input sequence and the dimension of the model
    OUTPUT: Encoded relative positions of the data points in the input sequence
    '''
    position = torch.arange(seq_length, dtype=torch.int).reshape(1, -1, 1).to(device)
    #position = torch.arange(seq_length, dtype=torch.float).reshape(1, -1, 1)
    frequencies = 1e-4 ** (2 * (torch.div(torch.arange(model_dim, dtype=torch.int), 2)) / model_dim).reshape(1, 1, -1).to(device)
    #frequencies = 1e-4 ** (2 * (torch.div(torch.arange(model_dim, dtype=torch.float), 2)) / model_dim).reshape(1, 1, -1)
    pos_enc = position * frequencies
    pos_enc[:, ::2] = torch.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = torch.sin(pos_enc[:, 1::2])
    return pos_enc

def forward(input_dim=512, forward_dim=2048):
    '''
    Forward class for the feed-forward layer that is following the multihead
    attention layers

    INPUT: input dimension and the layer size of the forward layer
    OUTPUT: feed-forward layer (nn.Module)
    '''
    forward_layer = nn.Sequential(
        nn.Linear(input_dim, forward_dim),
        nn.ReLU(),
        nn.Linear(forward_dim, input_dim)
    )
    return forward_layer
    
class ResidualConnection(nn.Module):
    '''
    Class for the residual connections for the encoder and the decoder, used for each multihead attention layer
    and for each feed-forward layer

    INPUT: type of layer, dimension for the layer normalization and dropout probability factor
    OUTPUT: Normalized and processed tensors added to the input tensors
    '''

    def __init__(self, layer, dimension, dropout=0.2):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *X):
        return self.norm(X[-1] + self.dropout(self.layer(*X)))
        
class Encoder(nn.Module):
    '''
    The encoder of the transformer model, first computes the relative positions of the inputs, then feeds it into
    the multihead attention followed by the feed-forward layer, both with normalized residual connections
    '''

    def __init__(self, device, n_layers=6, model_dim=512, num_heads=8, forward_dim=2048, dropout=0.2):
        super().__init__()
        
        self.device = device
        self.n_layers = n_layers
        key_dim = value_dim = model_dim // num_heads

        # Multihead attention layer with normalized residual connections and dropout
        self.multihead_attention = ResidualConnection(
            MultiHeadAttention(num_heads, model_dim, key_dim, value_dim),
            dimension=model_dim,
            dropout=dropout
        )
        # Feed-forward layer with normalized residual connections and dropout
        self.feed_forward = ResidualConnection(
            forward(model_dim, forward_dim),
            dimension=model_dim,
            dropout=dropout
        )

    def forward(self, X):
        seq_length, dimension = X.size(1), X.size(2)
        out = X
        # Computes the positional encodings
        out += positioning_encoding(self.device, seq_length, dimension)
        # Feeds the input to the multihead attention layer followed by the feed-forward
        # layer for 'n_layers' many layers
        for _ in range(self.n_layers):
            att_out = self.multihead_attention(out, out, out)
            out = self.feed_forward(att_out)
        return out
        
class TransformerModel(nn.Module):
    '''
    Transformer model that combines the encoder and the decoder
    "model_dim" must be the same size as "num_features" in the input data (i.e size last dimension),
    otherwise freely tunable parameters
    '''

    def __init__(self, device, n_layers=6, model_dim=512, output_dim=512,
                 num_heads=6, forward_dim=2048, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(device, n_layers, model_dim, num_heads, forward_dim, dropout)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4, output_dim)#todo why 4 is hardcoded?
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        enc_out = self.encoder(X)
        out = self.relu(self.linear(enc_out[:, -1, :]))
        return out
        
class TransformerModelImpl(nn.Module):
    def __init__(self, params):
        super(TransformerModelImpl, self).__init__()
        self.transf = TransformerModel(device = params.device,
                                       n_layers=params.n_layers,
                                       num_heads=params.num_heads,
                                       model_dim=params.model_dim,
                                       forward_dim=params.forward_dim,
                                       output_dim=4,#todo need to make this a parameter
                                       dropout=params.dropout)
        self.linear = nn.Linear(4, params.output_dim)#todo why to hardcode 4
    def forward(self, x):
        transf_out = self.transf(x)
        out = self.linear(transf_out)
        return out
        
#TransformerModelImpl(
#  (transf): TransformerModel(
#    (encoder): Encoder(
#      (multihead_attention): ResidualConnection(
#        (layer): MultiHeadAttention(
#          (query): Linear(in_features=4, out_features=4, bias=True)
#          (key): Linear(in_features=4, out_features=4, bias=True)
#          (value): Linear(in_features=4, out_features=4, bias=True)
#          (linear): Linear(in_features=4, out_features=4, bias=True)
#        )
#        (norm): LayerNorm((4,), eps=1e-05, elementwise_affine=True)
#        (dropout): Dropout(p=0.1, inplace=False)
#      )
#      (feed_forward): ResidualConnection(
#        (layer): Sequential(
#          (0): Linear(in_features=4, out_features=32, bias=True)
#          (1): ReLU()
#          (2): Linear(in_features=32, out_features=4, bias=True)
#        )
#        (norm): LayerNorm((4,), eps=1e-05, elementwise_affine=True)
#        (dropout): Dropout(p=0.1, inplace=False)
#      )
#    )
#    (flatten): Flatten(start_dim=1, end_dim=-1)
#    (linear): Linear(in_features=4, out_features=4, bias=True)
#    (relu): ReLU(inplace=True)
#  )
#  (linear): Linear(in_features=4, out_features=2, bias=True)
#)

class TransformerModelImpl2(nn.Module):
  def __init__(self,params):
    super(TransformerModelImpl2,self).__init__()
    self.device=params.device
    encoder_layer = nn.TransformerEncoderLayer(batch_first=True, dropout=params.dropout, d_model=params.d_model, nhead=params.nhead, dim_feedforward=params.dim_feedforward)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=params.num_layers)
    self.fc_layer = nn.Linear(params.d_model*params.seq_len,params.d_output)

  def forward(self,X):
      
    seq_length, dimension = X.size(1), X.size(2)
    out = X
    out += positioning_encoding(self.device, seq_length, dimension)
    out = self.transformer_encoder(out)
    out = torch.flatten(out,start_dim=1)   
    out = self.fc_layer(out)
    return out.reshape(X.shape[0],-1)
