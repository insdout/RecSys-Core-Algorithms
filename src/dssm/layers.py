import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class InputMask(nn.Module):
    """
    Module to create a mask based on non-padding elements in the input sequence.

    Attributes:
        None

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Generates a mask based on non-padding elements in the input sequence.

    """

    def __init__(self):
        super(InputMask, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to create a mask based on non-padding elements in the input sequence.

        Args:
            x (torch.Tensor): Input sequence.

        Returns:
            torch.Tensor: Mask generated based on non-padding elements.

        """
        mask = (x != 0).float()  # Assuming padding is 0 for simplicity
        return mask.unsqueeze(1).float()


class PoolingLayer(nn.Module):
    """
    Module to perform different types of pooling on input sequences.

    Attributes:
        pooling_type (str): Type of pooling operation ('concat', 'average', 'sum').

    Methods:
        forward(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            Performs the specified pooling operation on the input sequence.

    """

    def __init__(self, pooling_type: str = 'concat'):
        super(PoolingLayer, self).__init__()
        self.pooling_type = pooling_type

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward method to perform the specified pooling operation on the input sequence.

        Args:
            x (torch.Tensor): Input sequence.
            mask (torch.Tensor, optional): Mask for non-padding elements.

        Returns:
            torch.Tensor: Result of the pooling operation.

        """
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

    def average_pooling(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Performs average pooling on the input sequence.

        Args:
            x (torch.Tensor): Input sequence.
            mask (torch.Tensor): Mask for non-padding elements.

        Returns:
            torch.Tensor: Result of average pooling.

        """
        sum_pooling_matrix = torch.bmm(mask, x.float()).squeeze(1)
        non_padding_length = mask.sum(dim=-1)
        return sum_pooling_matrix / (non_padding_length.float() + 1e-16)

    def sum_pooling(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Performs sum pooling on the input sequence.

        Args:
            x (torch.Tensor): Input sequence.
            mask (torch.Tensor): Mask for non-padding elements.

        Returns:
            torch.Tensor: Result of sum pooling.

        """
        return torch.bmm(mask, x.float()).squeeze(1)


class Feature(object):
    """
    Class representing a feature in the model.

    Attributes:
        name (str): Name of the feature.
        vocab_dim (int): Vocabulary dimension of the feature.
        embedding_dim (int): Embedding dimension of the feature.
        sequential (bool): Whether the feature is sequential.
        padding (int): Padding value for the feature.
        pooling_type (str): Type of pooling operation for the feature.
        shared_embedding (Optional[str]): Name of the feature with a shared embedding.

    Methods:
        get_emb() -> nn.Embedding:
            Returns the embedding layer for the feature.

    """

    def __init__(
        self,
        name: str,
        vocab_dim: int,
        embedding_dim: int,
        sequential: bool = False,
        padding: int = 0,
        shared_embedding: Optional[str] = None,
        pooling_type: str = 'average',
    ):
        self.name = name
        self.vocab_dim = vocab_dim
        self.embedding_dim = embedding_dim
        self.sequential = sequential
        self.padding = padding
        self.pooling_type = pooling_type
        self.shared_embedding = shared_embedding
        self.emb = None

    def get_emb(self) -> nn.Embedding:
        """
        Returns the embedding layer for the feature.

        Returns:
            nn.Embedding: Embedding layer for the feature.

        """
        if not self.emb:
            self.emb = nn.Embedding(self.vocab_dim, self.embedding_dim, padding_idx=self.padding)
        return self.emb


class FeatureEmbeddings(nn.Module):
    """
    Module to handle multiple feature embeddings.

    Attributes:
        embedding_layers (nn.ModuleDict): Dictionary to store embedding layers for each feature.

    Methods:
        forward(x: dict, features: List[Feature]) -> torch.Tensor:
            Forward method to compute embeddings for multiple features.

    """

    def __init__(self, features: List[Feature]):
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

    def forward(self, x: dict, features: List[Feature]) -> torch.Tensor:
        """
        Forward method to compute embeddings for multiple features.

        Args:
            x (dict): Input dictionary containing feature values.
            features (List[Feature]): List of Feature objects.

        Returns:
            torch.Tensor: Concatenated embeddings for all features.

        """
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
    """
    Multilayer Perceptron (MLP) module.

    Attributes:
        mlp (nn.Sequential): Sequential container for MLP layers.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward method to compute the output of the MLP.

    """

    def __init__(
        self,
        input_dim: int,
        output_layer: bool = False,
        dims: Optional[List[int]] = None,
        dropout: float = 0,
        activation: str = "relu",
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to compute the output of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        out = self.mlp(x)
        out_norm = F.normalize(out, p=2, dim=1)
        return out_norm



def activation_layer(activation: str) -> nn.Module:
    """
    Creates an activation layer based on the specified activation function.

    Args:
        activation (str): Activation function name.

    Returns:
        nn.Module: Activation layer.

    Raises:
        ValueError: If the activation function is not supported.

    """
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

