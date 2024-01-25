import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dssm.layers import FeatureEmbeddings, MLP, Feature


class DSSMModel(nn.Module):
    def __init__(self, user_features, item_features, mlp_user_params, mlp_item_params, temperature):
        super(DSSMModel, self).__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.temperature = temperature
        self.feature_embeddings = FeatureEmbeddings(user_features + item_features)

        user_emb_dims = self._calculate_input_dims(self.user_features)
        item_emb_dims = self._calculate_input_dims(self.item_features)
        self.user_tower = MLP(user_emb_dims, mlp_user_params)
        self.item_tower = MLP(item_emb_dims, mlp_item_params)

    def _calculate_input_dims(self, features):
        return sum([feature.embedding_dim for feature in features])
    
    def forward(self, x):
        user_emb = self.feature_embeddings(x, self.user_features)
        item_emb = self.feature_embeddings(x, self.item_features)

        user_vec = self.user_tower(user_emb)
        item_vec = self.item_tower(item_emb)
        # Perform similarity computation (cosine similarity)
        
        y = (user_vec * item_vec).sum(dim=1) / self.temperature
        
        return torch.sigmoid(y)


if __name__ == '__main__':
    user_num = 1000
    item_num = 1000

    # Define Feature objects
    gender_feature = Feature(name='gender', vocab_dim=5, embedding_dim=8, sequential=False, padding=None)
    age_feature = Feature(name='age', vocab_dim=15, embedding_dim=8, sequential=False, padding=None)
    occupation_feature = Feature(name='occupation', vocab_dim=20, embedding_dim=8, sequential=False, padding=None)
    zip_feature = Feature(name='zip', vocab_dim=100, embedding_dim=8, sequential=False, padding=None)
    movie_id_feature = Feature(name='movie_id', vocab_dim=3000, embedding_dim=32, sequential=False, padding=None)
    hist_movie_id_feature = Feature(name='hist_movie_id', vocab_dim=1000, embedding_dim=32, sequential=True, padding=0, shared_embedding='movie_id')

 
    # Define MLP dimensions
    mlp_dims = [64, 32]

    temperature = 1.0

    # Create DSSMModel instance
    model = DSSMModel(user_features=[gender_feature, age_feature, occupation_feature, zip_feature],
                      item_features=[movie_id_feature, hist_movie_id_feature],
                      mlp_user_params={'dims': mlp_dims},
                      mlp_item_params={'dims': mlp_dims},
                      temperature=temperature)

    # Example input data
    data = {'user_id': torch.tensor([34, 17, 44]),
            'movie_id': torch.tensor([516, 409, 2027]),
            'hist_movie_id': torch.tensor([[653, 38, 2038, 1060, 670, 690, 644, 674, 481, 1121, 1726,
                                            647, 152, 700, 748, 720, 609, 668, 2065, 365, 897, 1570,
                                            693, 485, 1739, 472, 774, 477, 793, 799, 663, 276, 1076,
                                            649, 686, 63, 1065, 6, 729, 1618, 515, 1077, 688, 474,
                                            656, 437, 708, 355, 0, 0],
                                           [358, 1150, 220, 726, 119, 483, 359, 1155, 1162, 553, 558,
                                            564, 336, 1151, 562, 385, 353, 1148, 559, 867, 561, 577,
                                            141, 1536, 522, 935, 790, 595, 426, 470, 839, 560, 1111,
                                            442, 1156, 1994, 970, 37, 974, 1997, 703, 793, 1315, 2033,
                                            722, 1095, 757, 1936, 433, 0],
                                           [1894, 557, 945, 1802, 1372, 600, 1362, 146, 1257, 734, 1446,
                                            1203, 1136, 33, 2056, 1746, 1752, 570, 1285, 682, 678, 1,
                                            1671, 1742, 799, 722, 1309, 1342, 1873, 484, 1306, 1999, 729,
                                            1150, 1138, 587, 1619, 558, 823, 751, 1516, 584, 1015, 843,
                                            1341, 1616, 848, 1096, 1171, 909]]),
            'histlen_movie_id': torch.tensor([109, 396, 176]),
            'label': torch.tensor([0, 1, 0]),
            'gender': torch.tensor([2, 2, 1]),
            'age': torch.tensor([4, 3, 3]),
            'occupation': torch.tensor([8, 1, 3]),
            'zip': torch.tensor([29, 42, 39]),
            'cat_id': torch.tensor([8, 5, 8])}

    pred = model(data)
    print(pred)