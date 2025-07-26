import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout:  float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

        # Create a matrix of shape (seq_len, d_model)
        # row = seq_len
        # column = d_model
        pe = torch.zeros(seq_len, d_model)
        # Create a position matrix of shape (seq_len, d_model)

        # One torch.arange example:          One unsqueeze example:
        # torch.arange(5)                         x = tensor([0, 1, 2, 3, 4])
        # tensor([0, 1, 2, 3, 4])                 tensor.unsqueeze(x, 0)
        # torch.arange(1, 2.5, 0.5)               tensor([[0, 1, 2, 3, 4]])
        # tensor([1.0, 1.5, 2.0])                 tensor.unsqueeze(x, 1)
        #                                         tensor([[1],
        #                                                [2],
        #                                                [3],
        #                                                [4]])

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even indices in the array
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).require_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean  = x.mean(dim = -1, keepdim=True)
        std   = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.modules):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    # h: number of heads
    # d_model: dimension of the model
    def __init__(self, d_model: int, h:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h # dimension of each head
        self.w_q = nn.Linear(d_model, d_model)  # Wq queries
        self.w_k = nn.Linear(d_model, d_model)  # Wk keys
        self.w_v = nn.Linear(d_model, d_model)  # Wv values

        self.w_o = nn.Linear(d_model, d_model)  # Wo output
        self.dropout = nn.Dropout(dropout)
        
        @staticmethod
        def attention(query, key, value, mask, dropout):
            d_k = query.size(-1)  # dimension of each head

            # (batch, h, Seq_len, d_k) -> (Batch, h, Seq_len, Seq_len)
            attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # (Batch, h, Seq_len, Seq_len)
            # Apply the mask to the attention scores
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            attention_scores = torch.softmax(attention_scores, dim=-1)  # (Batch, h, Seq_len, Seq_len)
            if dropout is not None:
                attention_scores = dropout(attention_scores)

            return (attention_scores @ value), attention_scores




        def forward(self, q, k, v, mask):
            # w_q(q) means "apply the linear transformation w_q to q
            query = self.w_q(q)  # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
            key = self.w_k(k)  # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
            value = self.w_v(v)  # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)


            # (batch, Seq_len, d_model) -> (Batch, Seq_len, h, d_k) -> (Batch, h, Seq_len, d_k)
            query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2)  # (Batch, h, Seq_len, d_k)
            key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)
            value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2)

            x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
            # (batch, h, Seq_len, d_k) -> (Batch, Seq_len, h, d_k) -> (Batch, Seq_len, d_model)
            x = x.transpose(1,2).contiguous().view(x.size(0),-1, self.h*self.d_k)  # (Batch, Seq_len, d_model)

            return self.w_o(x)  # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)