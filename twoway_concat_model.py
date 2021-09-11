import os
import pickle
import time
import numpy as np
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F


class twoway_concat(nn.Module):
    def __init__(self, item_idx_tensor, item_scr_tensor,
                 user_idx_tensor, user_scr_tensor,
                 data_dir, n_items, n_users, emb_dim, multi_factor, top_k):
        super(twoway_concat, self).__init__()

        self.item_idx_tensor = item_idx_tensor
        self.item_scr_tensor = item_scr_tensor
        # torch.Size([3515, multi_factor x # of neighbors])
        self.user_idx_tensor = user_idx_tensor
        self.user_scr_tensor = user_scr_tensor
        # torch.Size([6034, multi_factor x # of neighbors])

        self.multi_factor = multi_factor
        self.emb_dim = emb_dim

        self.user_emb = nn.Embedding(n_users, emb_dim).to('cuda')
        self.item_emb = nn.Embedding(n_items, emb_dim).to('cuda')

        self.concat_net_1 = concat_linear(input_dim=multi_factor * top_k * 2,
                                          output_dim=1).to('cuda')
        self.concat_net_2 = concat_linear(input_dim=multi_factor * top_k * 2,
                                          output_dim=1).to('cuda')

    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs]
        # torch.Size([3515, 100 = (multi_factor x # of neighbors)])
        user_neigh_emb = self.user_emb(user_neighs)
        # torch.Size([3515, 100, 64])
        user_neigh_scr = self.user_scr_tensor[user_idxs]
        # torch.Size([3515, 100])
        user_neigh_scr = user_neigh_scr[:, np.newaxis, :].repeat(1, self.emb_dim, 1)
        # torch.Size([3515, 64, 100])
        scored_user_emb = torch.bmm(user_neigh_emb, user_neigh_scr)
        # [3515, 100, 100]

        item_neighs = self.item_idx_tensor[item_idxs]
        item_neigh_emb = self.item_emb(item_neighs)
        item_neigh_scr = self.item_scr_tensor[item_idxs]
        item_neigh_scr = item_neigh_scr[:, np.newaxis, :].repeat(1, self.emb_dim, 1)
        scored_item_emb = torch.bmm(item_neigh_emb, item_neigh_scr)
        # [3515, 100, 100]

        # print('concat 1 : ', torch.cat((scored_user_emb, scored_item_emb), 1).shape)
        # torch.Size([3515, 200, 100])
        # print('concat 2 : ', torch.cat((scored_user_emb, scored_item_emb), 2).shape)
        # torch.Size([3515, 100, 200])
        user_item_emb_1 = torch.cat((scored_user_emb, scored_item_emb), 1)
        user_item_emb_2 = torch.cat((scored_user_emb, scored_item_emb), 2)
        concat_1 = self.concat_net_1(user_item_emb_1.permute(0, 2, 1))
        concat_2 = self.concat_net_2(user_item_emb_2)
        # torch.Size([3515, 100, 1])
        results = torch.sigmoid(torch.mean(concat_1.squeeze() + concat_2.squeeze(), dim=1))
        return results


class concat_linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(concat_linear, self).__init__()

        self.ln1 = nn.Linear(input_dim, input_dim//2)
        self.ln2 = nn.Linear(input_dim//2, input_dim//4)
        self.ln3 = nn.Linear(input_dim//4, input_dim//8)
        self.ln4 = nn.Linear(input_dim//8, input_dim//16)
        self.ln5 = nn.Linear(input_dim//16, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = self.ln5(x)
        return x
