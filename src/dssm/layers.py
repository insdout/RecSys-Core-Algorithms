import torch
import torch.nn as nn
import torch.nn.functional as F

class InputMask(nn.Module):
    def __init__(self):
        super(InputMask, self).__init__()

    def forward(self, x):
        mask = (x != 0).float()  # Assuming padding is 0 for simplicity
        return mask.unsqueeze(1).float()

class PoolingLayer(nn.Module):
    def __init__(self, pooling_type='concat'):
        super(PoolingLayer, self).__init__()
        self.pooling_type = pooling_type

    def forward(self, x, mask=None):
        if self.pooling_type == 'concat':
            return x
        elif self.pooling_type == 'average':
            if mask is None:
                mask = (x != 0).float()  # Assuming padding is 0 for simplicity
            return self.average_pooling(x, mask)
        elif self.pooling_type == 'sum':
            if mask is None:
                mask = (x != 0).float()  # Assuming padding is 0 for simplicity
            return self.sum_pooling(x, mask)
        else:
            raise ValueError("Invalid pooling type. Supported types: 'concat', 'average', 'sum'")

    def average_pooling(self, x, mask):
        sum_pooling_matrix = torch.bmm(mask, x.float()).squeeze(1)
        non_padding_length = mask.sum(dim=-1)
        return sum_pooling_matrix / (non_padding_length.float() + 1e-16)

    def sum_pooling(self, x, mask):
        return torch.bmm(mask, x.float()).squeeze(1)


class Feature(object):
    def __init__(self, name, vocab_dim, embedding_dim, sequential=False, padding=0, shared_embedding=None, pooling_type='average'):
        self.name = name
        self.vocab_dim = vocab_dim
        self.embedding_dim = embedding_dim
        self.sequential = sequential
        self.padding = padding
        self.pooling_type = pooling_type
        self.shared_embedding = shared_embedding

    def get_emb(self):
        return nn.Embedding(self.vocab_dim, self.embedding_dim, padding_idx=self.padding)


class FeatureEmbeddings(nn.Module):
    def __init__(self, features):
        super(FeatureEmbeddings, self).__init__()

        self.embedding_layers = nn.ModuleDict()

        for feature in features:
            feature_name = feature.name
            shared_embedding = feature.shared_embedding
            if not shared_embedding:
                self.embedding_layers[feature_name] = feature.get_emb()
        for feature in features:
            feature_name = feature.name
            if feature_name not in self.embedding_layers:
                shared_embedding = feature.shared_embedding
                if shared_embedding:
                    self.embedding_layers[feature_name] = self.embedding_layers[shared_embedding]
        

    def forward(self, x, features):
        embeddings = []
        for feature in features:
            feature_name = feature.name
            sequential = feature.sequential
            pooling_type = feature.pooling_type
            x_in = x[feature_name]
            if sequential:
                mask = InputMask()(x_in)
                emb_seq = self.embedding_layers[feature_name](x_in)
                emb = PoolingLayer(pooling_type=pooling_type)(emb_seq, mask)
            else:
                emb = self.embedding_layers[feature_name](x_in).squeeze(1)
            embeddings.append(emb)
        out = torch.cat(embeddings, dim=-1)  # Concatenate along the last dimension
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
        super(MLP, self).__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        out_norm = F.normalize(out, p=2, dim=1)
        return out_norm

# Activation layer function
def activation_layer(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softmax':
        return nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Invalid activation function: {activation}")
