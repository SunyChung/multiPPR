import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_features import factorFeatures
from utils import load_all


class FactorCNNEmb(nn.Module):
    def __init__(self, factor_item_idx_tensor, factor_item_scr_tensor,
                 factor_user_idx_tensor, factor_user_scr_tensor,
                 item_embedding, user_embedding,
                 n_items, top_k, multi_factor, emb_dim):
        super(FactorCNNEmb, self).__init__()

        self.item_idx_tensor = factor_item_idx_tensor
        # torch.Size([3515, 5, 20])
        self.item_scr_tensor = factor_item_scr_tensor
        self.user_idx_tensor = factor_user_idx_tensor
        # torch.Size([6034, 5, 20])
        self.user_scr_tensor = factor_user_scr_tensor
        self.n_items = n_items
        self.top_k = top_k
        self.multi_factor = multi_factor
        self.emb_dim = emb_dim

        self.item_emb = item_embedding
        self.user_emb = user_embedding

        self.c_net = EmbConNet(self.emb_dim, chan1=10, chan2=5)
        # self.c_net = FactorConNet(self.multi_factor, chan1=10, chan2=5)

    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs, :, :]
        # torch.Size([3515, 5, 20])
        user_neigh_emb = self.user_emb(user_neighs)
        # torch.Size([3515, 5, 20, 64])
        user_neigh_scr = self.user_scr_tensor[user_idxs, :, :]
        # torch.Size([3515, 5, 20])
        user_neigh_scr = user_neigh_scr[:, :, np.newaxis, :,].repeat(1, 1, self.emb_dim,1)
        # torch.Size([3515, 5, 64, 20])
        user_factor_tensor = torch.stack(
            [torch.sum(torch.bmm(user_neigh_scr[:, i, :, :], user_neigh_emb[:, i, :, :]), dim=2)
             for i in range(self.multi_factor)], dim=1)
        # torch.Size([3515, 5, 64])

        item_neighs = self.item_idx_tensor[item_idxs, :, :]
        item_neigh_emb = self.item_emb(item_neighs)
        item_neigh_scr = self.item_scr_tensor[item_idxs, :, :]
        item_neigh_scr = item_neigh_scr[:, :, np.newaxis, :].repeat(1, 1, self.emb_dim,1)
        item_factor_tensor = torch.stack(
            [torch.sum(torch.bmm(item_neigh_scr[:, i, :, :], item_neigh_emb[:, i, :, :]), dim=2)
             for i in range(self.multi_factor)], dim=1)
        # torch.Size([3515, 5, 64])

        # input [5, 5] -> FactorConNet
        # user_item = torch.bmm(user_factor_tensor, torch.transpose(item_factor_tensor, 1, 2))
        # torch.Size([3515, 5, 5])

        # OR [64, 64] -> EmbConNet
        user_item = torch.bmm(torch.transpose(item_factor_tensor, 1, 2), user_factor_tensor)
        # torch.Size([3515, 64, 64])

        result = torch.sigmoid(self.c_net(user_item.unsqueeze(1)))
        return result.squeeze()


class EmbConNet(nn.Module):
    def __init__(self, emb_dim, chan1, chan2):
        super(EmbConNet, self).__init__()
        self.conv1 = nn.Conv2d(1, chan1, emb_dim // 2)
        self.conv2 = nn.Conv2d(chan1, chan2, emb_dim // 4)
        last_filter = emb_dim - emb_dim // 2 - emb_dim // 4 + 2
        linear_in = chan2 * last_filter * last_filter
        self.fc1 = nn.Linear(linear_in, linear_in // 2)
        self.fc2 = nn.Linear(linear_in // 2, linear_in // 4)
        self.fc3 = nn.Linear(linear_in // 4, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # torch.Size([3515, 10, 33, 33])
        x = F.relu(self.conv2(x))
        # torch.Size([3515, 5, 18, 18])
        x = x.view(x.size(0), -1)
        # print('x shape : ', x.shape)  # torch.Size([3515, 1620])
        # 여기서 flatten() 쓰면 batch 가 다 풀어서 곱해지니,
        # batch_size 를 살리려면, flatten() 쓰지 말 것 !!
        # x = torch.flatten(x)
        # print('x shape : ', x.shape)
        x = F.relu(self.fc1(x))
        # print('x shape : ', x.shape) # torch.Size([3515, 810])
        x = F.relu(self.fc2(x))
        # print('x shape : ', x.shape)  # torch.Size([3515, 405])
        x = F.relu(self.fc3(x))
        # print('x shape : ', x.shape)  # torch.Size([3515, 1])
        return x


class FactorConNet(nn.Module):
    def __init__(self, fac_num, chan1, chan2):
        super(FactorConNet, self).__init__()
        self.conv1 = nn.Conv2d(1, chan1, 3)
        self.conv2 = nn.Conv2d(chan1, chan2, 1)
        last_filter = fac_num - 3 - 1 + 2
        linear_in = chan2 * last_filter * last_filter
        self.fc1 = nn.Linear(linear_in, linear_in // 2)
        self.fc2 = nn.Linear(linear_in // 2, linear_in // 4)
        self.fc3 = nn.Linear(linear_in // 4, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # torch.Size([3515, 10, 3, 3])
        x = F.relu(self.conv2(x))
        # torch.Size([3515, 5, 3, 3])
        x = x.view(x.size(0), -1)
        # torch.Size([3515, 45])
        x = F.relu(self.fc1(x))
        # torch.Size([3515, 22])
        x = F.relu(self.fc2(x))
        # torch.Size([3515, 11])
        x = F.relu(self.fc3(x))
         # torch.Size([3515, 1])
        return x


if __name__ == '__main__':
    data_dir = './data/ml-1m'
    multi_factor = 5
    top_k = 20

    # item feature tensor
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
        item_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
        item_ppr_dict = pickle.load(f)
    item_ff = factorFeatures(multi_factor, top_k, item_idx_dict, item_ppr_dict)
    factor_item_idx_tensor = item_ff.idx_multi_tensors()
    factor_item_scr_tensor = item_ff.scr_multi_tensors()
    del item_idx_dict
    del item_ppr_dict

    # user feature tensor
    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
        user_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
        user_ppr_dict = pickle.load(f)
    user_ff = factorFeatures(multi_factor, top_k, user_idx_dict, user_ppr_dict)
    factor_user_idx_tensor = user_ff.idx_multi_tensors()
    factor_user_scr_tensor = user_ff.scr_multi_tensors()
    del user_idx_dict
    del user_ppr_dict

    multi_factor = 5
    emb_dim = 64
    n_items, train_data, vad_data, test_data = load_all(data_dir)
    batch_size = 100

    with open(os.path.join(data_dir, 'unique_uidx'), 'rb') as f:
        unique_uidx = pickle.load(f)
    item_embedding = nn.Embedding(n_items, emb_dim)
    user_embedding = nn.Embedding(len(unique_uidx), emb_dim)

    femb = FactorCNNEmb(factor_item_idx_tensor, factor_item_scr_tensor,
                      factor_user_idx_tensor, factor_user_scr_tensor,
                      item_embedding, user_embedding,
                      n_items, top_k, multi_factor, emb_dim)
    train_n = len(train_data)
    train_idxlist = list(range(train_n))
    loss_list = []
    for batch_num, st_idx in enumerate(range(0, train_n, batch_size)):
        end_idx = min(st_idx + batch_size, train_n)
        user_idxs = train_data[train_idxlist[st_idx:end_idx]][:, 0]
        item_idxs = train_data[train_idxlist[st_idx:end_idx]][:, 1]
        result = femb(user_idxs, item_idxs)
        print('result : ', result)
