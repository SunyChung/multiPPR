import os
import pickle
import numpy as np
from features import PPRfeatures
from utils import load_all
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmb(object):
    def __init__(self, item_idx_tensor, item_scr_tensor,
                 user_idx_tensor, user_scr_tensor,
                 item_embedding, user_embedding,
                 n_items, top_k, multi_factor, emb_dim):
        super(FeatureEmb, self).__init__()

        self.item_idx_tensor = item_idx_tensor  # torch.Size([3515, multi_factor x # of neighbors])
        self.item_scr_tensor = item_scr_tensor
        self.user_idx_tensor = user_idx_tensor  # torch.Size([6034, multi_factor x # of neighbors])
        self.user_scr_tensor = user_scr_tensor
        self.n_items = n_items
        self.top_k = top_k
        self.multi_factor = multi_factor
        self.emb_dim = emb_dim

        # print('\ndefault weight initialization')
        # print('\nusing xavier_normal_ weight initialization')
        print('\n kaiming_normal_ weight initialization')

        self.item_emb = item_embedding
        # print('item_emb : ', self.item_emb.weight)
        print('item_emb min : ', torch.min(self.item_emb.weight))
        print('item_emb mean : ', torch.mean(self.item_emb.weight))
        print('item_emb std : ', torch.std(self.item_emb.weight))
        print('item_emb max : ', torch.max(self.item_emb.weight))
        # nn.init.xavier_normal_(self.item_emb.weight)
        # print('item_emb initialized : ', self.item_emb.weight)
        nn.init.kaiming_normal_(self.item_emb.weight)

        self.user_emb = user_embedding
        # print('user_emb : ', self.user_emb.weight)
        print('\nuser_emb min : ', torch.min(self.user_emb.weight))
        print('user_emb mean : ', torch.mean(self.user_emb.weight))
        print('user_emb std : ', torch.std(self.user_emb.weight))
        print('user_emb max : ', torch.max(self.user_emb.weight))
        # nn.init.xavier_normal_(self.user_emb.weight)
        # print('user_emb initialized : ', self.user_emb.weight)
        nn.init.kaiming_normal_(self.user_emb.weight)

        self.c_net = ConvNet(chan_1=64, chan_2=32)

    def forward(self, user_idxs, item_idxs):
        # print('user_idxs shape : ', user_idxs.shape)  # (3516,)
        user_neighs = self.user_idx_tensor[user_idxs]
        print('\nuser_neighs shape : ', user_neighs.shape)
        # torch.Size([3516(=batch_size=# of items), multi_factor x top_k])
        neigh_emb = self.user_emb(user_neighs)
        print('neigh_emb shape : ', neigh_emb.shape)
        # torch.Size([3516(=batch_size), multi_factor x top_k, embedding_dim])
        neigh_score = self.user_scr_tensor[user_neighs]
        print('neigh_score shape : ', neigh_score.shape)
        # torch.Size([3516(=batch_size), multi_factor x top_k, multi_factor x top_k])
        scored_user_emb = torch.bmm(neigh_score, neigh_emb)
        print('scored_user_emb shape : ', scored_user_emb.shape)
        # torch.Size([3516(=batch_size), multi_factor x top_k, embedding_dim])
        summed_user = {}
        for i in range(self.multi_factor):
            # print('before sum shape : ', scored_user_emb[:, self.top_k * i: self.top_k * (i+1), :].shape)
            # torch.Size([3516, 20, 64])
            summed_user[i] = torch.sum(scored_user_emb[:, self.top_k * i: self.top_k * (i+1), :], dim=1)
            # print('summed_user shape: ', summed_user[i].shape)  # torch.Size([3516, 64])
            # print('summed_user : ', summed_user[i])

        concat_user_emb = torch.cat(list(summed_user.values()), dim=1)
        stacked_user_emb = torch.stack(list(summed_user.values()), dim=1)
        # concat keep the given dimension
        print('\ncon_user_emb shape : ', concat_user_emb.shape)  # torch.Size([3516, 320])
        # stack makes a new dimension
        print('stacked_user_emb shape : ', stacked_user_emb.shape)  # torch.Size([3516, 5, 64])

        item_neighs = self.item_idx_tensor[item_idxs]
        item_neigh_emb = self.item_emb(item_neighs)
        item_neigh_scr = self.item_scr_tensor[item_neighs]
        scored_item_emb = torch.bmm(item_neigh_scr, item_neigh_emb)
        summed_item = {}
        for i in range(self.multi_factor):
            summed_item[i] = torch.sum(scored_item_emb[:, self.top_k * i: self.top_k * (i+1), :], dim=1)
        concat_item_emb = torch.cat(list(summed_item.values()), dim=1)  # torch.Size([17580, 64])
        stacked_item_emb = torch.stack(list(summed_item.values()), dim=1)  # torch.Size([3516, 5, 64])

        concat_user_item = torch.mm(concat_user_emb, concat_item_emb.T)
        # print('concat_user_item shape : ', concat_user_item.shape)  # torch.Size([17580, 17580])
        stacked_user_item = torch.bmm(stacked_user_emb, stacked_item_emb.permute(0, 2, 1))
        # print('stacked_user_item shape : ', stacked_user_item.shape)  # torch.Size([3516, 5, 5])
        # print('len(stacked_user_item) : ', len(stacked_user_item))

        results = []
        for i in range(len(stacked_user_item)):
            # print('input shape : ',
            #       stacked_user_item[i, :, :].unsqueeze_(0).unsqueeze_(0).shape)
            #  torch.Size([1, 1, 5, 5])
            result = torch.sigmoid(
                self.c_net(stacked_user_item[i, :, :].unsqueeze_(0).unsqueeze_(0)))
            # print('result shape : ', result.shape)
            # torch.Size([1])
            results.append(result)
        return results


class ConvNet(nn.Module):
    def __init__(self, chan_1=64, chan_2=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, chan_1, (1, 5))
        self.conv2 = nn.Conv2d(chan_1, chan_2, (5, 1))
        self.fc1 = nn.Linear(chan_2, chan_2 // 2)
        self.fc2 = nn.Linear(chan_2 // 2, chan_2 // 4)
        self.fc3 = nn.Linear(chan_2 // 4, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print('x shape : ', x.shape)
        x = F.relu(self.conv2(x))
        # print('x shape : ', x.shape)
        x = torch.flatten(x)
        # print('x shape : ', x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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
    femb = FeatureEmb(item_idx_tensor, item_scr_tensor,
                      user_idx_tensor, user_scr_tensor,
                      item_embedding, user_embedding,
                      n_items, top_k, multi_factor, emb_dim)

    results \
        = femb.forward(np.random.randint(0, 6000, 3516), np.random.randint(0, 3000, 3516))
    print(results)
