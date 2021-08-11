import os
import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import PPRfeatures
from utils import load_all


class ContextualizedNN(nn.Module):
    def __init__(self, item_idx_tensor, item_scr_tensor,
                 user_idx_tensor, user_scr_tensor,
                 item_embedding, user_embedding,
                 multi_factor, top_k):
        super(ContextualizedNN, self).__init__()

        self.item_idx_tensor = item_idx_tensor  # torch.Size([3515, multi_factor x # of neighbors])
        self.item_scr_tensor = item_scr_tensor
        self.user_idx_tensor = user_idx_tensor  # torch.Size([6034, multi_factor x # of neighbors])
        self.user_scr_tensor = user_scr_tensor

        # print('\ndefault weight initialization')
        print('\nusing xavier_normal_ weight initialization')
        # print('\n kaiming_normal_ weight initialization')

        self.item_emb = item_embedding
        # print('item_emb : ', self.item_emb.weight)
        print('item_emb min : ', torch.min(self.item_emb.weight))
        print('item_emb mean : ', torch.mean(self.item_emb.weight))
        print('item_emb std : ', torch.std(self.item_emb.weight))
        print('item_emb max : ', torch.max(self.item_emb.weight))
        nn.init.xavier_normal_(self.item_emb.weight)
        # print('item_emb initialized : ', self.item_emb.weight)
        # nn.init.kaiming_normal_(self.item_emb.weight)

        self.user_emb = user_embedding
        # print('user_emb : ', self.user_emb.weight)
        print('\nuser_emb min : ', torch.min(self.user_emb.weight))
        print('user_emb mean : ', torch.mean(self.user_emb.weight))
        print('user_emb std : ', torch.std(self.user_emb.weight))
        print('user_emb max : ', torch.max(self.user_emb.weight))
        nn.init.xavier_normal_(self.user_emb.weight)
        # print('user_emb initialized : ', self.user_emb.weight)
        # nn.init.kaiming_normal_(self.user_emb.weight)

        # self.cat_dim = multi_factor * top_k * 2
        self.input_dim = multi_factor * top_k

        # self.inter_layer = InterLin_1(input_dim=self.cat_dim, output_dim=1)

        # self.inter_layer = InterLin_3(input_dim=self.cat_dim,
        #                          hidden1=self.cat_dim // 2,
        #                          hidden2=self.cat_dim // 4,
        #                          output_dim=1)

        self.inter_layer = InterLin_5(input_dim=self.input_dim,
                                       hidden1=self.input_dim // 2,
                                       hidden2=self.input_dim // 4,
                                       output_dim=1)

        # self.inter_layer = InterLin_5_no_relu(input_dim=self.cat_dim,
        #                                     hidden1=self.cat_dim // 2,
        #                                     hidden2=self.cat_dim // 4,
        #                                     output_dim=1)

        # self.inter_layer = InterLin_7(input_dim=self.input_dim,
        #                               hidden1=self.input_dim // 2,
        #                               hidden2=self.input_dim // 4,
        #                               hidden3=self.input_dim // 8,
        #                               output_dim=1)

        # self.inter_layer = InterLin_8(input_dim=self.cat_dim,
        #                               hidden1=self.cat_dim // 2,
        #                               hidden2=self.cat_dim // 4,
        #                               hidden3=self.cat_dim // 8,
        #                               output_dim=1)

        # self.inter_layer = InterLin_12(input_dim=self.cat_dim,
        #                                 hidden1=self.cat_dim // 2,
        #                                 hidden2=self.cat_dim // 4,
        #                                 hidden3=self.cat_dim // 8,
        #                                 output_dim=1)

    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs]
        # print('user_neighs shape ', user_neighs.shape)
        # torch.Size([3515(=batch_size=# of items), multi_factor x # of neighbors])
        neigh_emb = self.user_emb(user_neighs)
        # print('neigh_emb shape : ', neigh_emb.shape)
        # torch.Size([3515(=batch_size), multi_factor x # of neighbors, embedding_dim])
        # neigh_score = self.user_scr_tensor[user_neighs]
        neigh_score = self.user_scr_tensor[user_idxs]
        # print('neigh_score : ', neigh_score)
        # print('neigh_score shape : ', neigh_score.shape)
        # torch.Size([3515(=batch_size), multi_factor x # of neighbors])
        neigh_score = neigh_score.unsqueeze(1).repeat(1, self.user_emb.embedding_dim, 1)
        # print('repeated neigh_score : ', neigh_score)
        # print('repeated neigh_score shape : ', neigh_score.shape)
        # torch.Size([3515(=batch_size), embedding_dim, multi_factor x # of neighbors])
        scored_user_emb = torch.bmm(neigh_emb, neigh_score)
        # print('scored_user_emb shape : ', scored_user_emb.shape)
        # torch.Size([3515(=batch_size), multi_factor x # of neighbors, multi_factor x # of neighbors])
        # torch.Size([3515, 100, 100])

        item_neighs = self.item_idx_tensor[item_idxs]
        item_neigh_emb = self.item_emb(item_neighs)
        item_neigh_scr = self.item_scr_tensor[item_idxs]
        item_neigh_scr = item_neigh_scr.unsqueeze(1).repeat(1, self.item_emb.embedding_dim, 1)
        scored_item_emb = torch.bmm(item_neigh_emb, item_neigh_scr)
        
        # print('inter cat shape : ', torch.cat((scored_user_emb, scored_item_emb), 2).shape)
        # torch.Size([3515, 100, 200])
        # 100 or 150 depends on the 'multi_factor x top_k' values !
        # user-item 벡터 concatenate
        # interaction_cat = torch.cat((scored_user_emb, scored_item_emb), 2)
        # print('interaction_cat shape : ', interaction_cat.shape)

        # concatenate 말고, 곱해서 넣어 보자 ... ;
        user_item_emb = torch.bmm(scored_user_emb, scored_item_emb)
        # print('user_item_emb shape : ', user_item_emb.shape)

        # with BCELoss()
        # result = torch.sigmoid(self.inter_layer(interaction_cat))
        result = torch.sigmoid(self.inter_layer(user_item_emb))
        # print('result shape : ', result.shape)  # torch.Size([3515, 150, 1])
        # print('return shape : ', torch.mean(result, dim=0).shape)  # torch.Size([150, 1])
        # print('return shape : ', torch.mean(result, dim=1).shape)  # torch.Size([3515, 1])
        # print('return shape : ', torch.mean(result, dim=-2).squeeze().shape)  # torch.Size([3515])
        # print('result : ', result)
        # print('result shape : ', result.shape)
        return torch.mean(result, dim=-2).squeeze()


class InterLin_1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InterLin_1, self).__init__()

        self.ln = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x = F.relu(self.ln(x))
        x = self.ln(x)
        return x


class InterLin_3(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(InterLin_3, self).__init__()

        self.ln1 = nn.Linear(input_dim, hidden1)
        self.ln2 = nn.Linear(hidden1, hidden2)
        self.ln3 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return x


class InterLin_5(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(InterLin_5, self).__init__()

        self.ln1 = nn.Linear(input_dim, hidden1)
        self.ln2 = nn.Linear(hidden1, hidden1)
        self.ln3 = nn.Linear(hidden1, hidden2)
        self.ln4 = nn.Linear(hidden2, hidden2)
        self.ln5 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = self.ln5(x)
        return x


# relu() 없으면 prediction 이 minus 값으로 나옴 -_;
# NO! sigmoid() 처리 때문임!!!
# class InterLin_5_no_relu(nn.Module):
#     def __init__(self, input_dim, hidden1, hidden2, output_dim):
#         super(InterLin_5_no_relu, self).__init__()
#
#         self.ln1 = nn.Linear(input_dim, hidden1)
#         self.ln2 = nn.Linear(hidden1, hidden1)
#         self.ln3 = nn.Linear(hidden1, hidden2)
#         self.ln4 = nn.Linear(hidden2, hidden2)
#         self.ln5 = nn.Linear(hidden2, output_dim)
#
#     def forward(self, x):
#         x = self.ln1(x)
#         x = self.ln2(x)
#         x = self.ln3(x)
#         x = self.ln4(x)
#         x = self.ln5(x)
#         return x


class InterLin_7(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3, output_dim):
        super(InterLin_7, self).__init__()

        self.ln1 = nn.Linear(input_dim, input_dim)
        self.ln2 = nn.Linear(input_dim, hidden1)
        self.ln3 = nn.Linear(hidden1, hidden2)
        self.ln4 = nn.Linear(hidden2, hidden2)
        self.ln5 = nn.Linear(hidden2, hidden3)
        self.ln6 = nn.Linear(hidden3, hidden3)
        self.ln7 = nn.Linear(hidden3, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = F.relu(self.ln5(x))
        x = F.relu(self.ln6(x))
        x = self.ln7(x)
        return x


class InterLin_8(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3, output_dim):
        super(InterLin_8, self).__init__()

        self.ln1 = nn.Linear(input_dim, input_dim)
        self.ln2 = nn.Linear(input_dim, hidden1)
        self.ln3 = nn.Linear(hidden1, hidden1)
        self.ln4 = nn.Linear(hidden1, hidden2)
        self.ln5 = nn.Linear(hidden2, hidden2)
        self.ln6 = nn.Linear(hidden2, hidden3)
        self.ln7 = nn.Linear(hidden3, hidden3)
        self.ln8 = nn.Linear(hidden3, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = F.relu(self.ln5(x))
        x = F.relu(self.ln6(x))
        x = F.relu(self.ln7(x))
        x = self.ln8(x)
        return x


class InterLin_12(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3, output_dim):
        super(InterLin_12, self).__init__()

        self.ln1 = nn.Linear(input_dim, input_dim)
        self.ln2 = nn.Linear(input_dim, input_dim)
        self.ln3 = nn.Linear(input_dim, hidden1)
        self.ln4 = nn.Linear(hidden1, hidden1)
        self.ln5 = nn.Linear(hidden1, hidden1)
        self.ln6 = nn.Linear(hidden1, hidden2)
        self.ln7 = nn.Linear(hidden2, hidden2)
        self.ln8 = nn.Linear(hidden2, hidden2)
        self.ln9 = nn.Linear(hidden2, hidden3)
        self.ln10 = nn.Linear(hidden3, hidden3)
        self.ln11 = nn.Linear(hidden3, hidden3)
        self.ln12 = nn.Linear(hidden3, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = F.relu(self.ln5(x))
        x = F.relu(self.ln6(x))
        x = F.relu(self.ln7(x))
        x = F.relu(self.ln8(x))
        x = F.relu(self.ln9(x))
        x = F.relu(self.ln10(x))
        x = F.relu(self.ln11(x))
        x = self.ln12(x)
        return x


if __name__ == '__main__':
    data_dir = './data/ml-1m'
    top_k = 20

    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
        item_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
        item_ppr_dict = pickle.load(f)
    pf = PPRfeatures(top_k, item_idx_dict, item_ppr_dict)
    item_idx_tensor = pf.idx_tensor()
    item_scr_tensor = pf.scr_tensor()
    del item_idx_dict
    del item_ppr_dict

    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
        user_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
        user_ppr_dict = pickle.load(f)
    pf = PPRfeatures(top_k, user_idx_dict, user_ppr_dict)
    user_idx_tensor = pf.idx_tensor()
    user_scr_tensor = pf.scr_tensor()
    del user_idx_dict
    del user_ppr_dict

    multi_factor = 5
    emb_dim = 64
    n_items, train_data, vad_data, test_data = load_all(data_dir)
    with open(os.path.join(data_dir, 'unique_uidx'), 'rb') as f:
        unique_uidx = pickle.load(f)
    item_embedding = nn.Embedding(n_items, emb_dim)
    user_embedding = nn.Embedding(len(unique_uidx), emb_dim)

    model = ContextualizedNN(item_idx_tensor, item_scr_tensor,
                             user_idx_tensor, user_scr_tensor,
                             item_embedding, user_embedding,
                             multi_factor, top_k)

    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    ex_idxs = train.loc[train['userId'] == 0].to_numpy()
    result = model(ex_idxs[:, 0], ex_idxs[:, 1])
