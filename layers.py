import os
import pickle
import numpy as np
import torch
import torch.nn as nn


class PPRfeatures(object):
    def __init__(self, data_dir, top_k, idx_dict, ppr_dict):
        self.data_dir = data_dir
        self.top_k = top_k
        self.idx_values = np.array(list(idx_dict.values()))
        self.ppr_scores = np.array(list(ppr_dict.values()))

    def idx_tensor(self):
        # dictionary shape : [all nodes idxs , multi-factors, ranked idxs]
        idx_top_k = self.idx_values[:, :, :self.top_k]
        idx_tensor = torch.LongTensor(idx_top_k.reshape(idx_top_k.shape[0], -1))
        # print('idx_tensor shape : ', idx_tensor.shape)  # torch.Size([3505, 100])
        return idx_tensor

    def score_tensor(self):
        # dictionary shape : [all nodes idxs, multi-factors, ranked scores]
        score_top_k = self.ppr_scores[:, :, :self.top_k]
        score_tensor = torch.FloatTensor(score_top_k.reshape(score_top_k.shape[0], -1))
        # print('score_tensor shape : ', score_tensor.shape)  # torch.Size([3505, 100])
        return score_tensor

    # previous try with nn.Embedding
    # idx_embeds got torch.FloatTensor(Tensor) as the dtype,
    # thus, can not be used as the another input to the nn.Embedding
    def idx_embeds(self):
        idx_top_k = self.idx_values[:, :, 1:self.top_k+1]
        idx_emb = nn.Embedding.from_pretrained(
            torch.Tensor(idx_top_k.reshape(idx_top_k.shape[0], -1)), freeze=True)
        return idx_emb

    def score_embeds(self):
        score_top_k = self.ppr_scores[:, :, 1:self.top_k+1]
        score_emb = nn.Embedding.from_pretrained(
            torch.Tensor(score_top_k.reshape(score_top_k.shape[0], -1)), freeze=True)
        return score_emb


if __name__ == '__main__':
    # data_dir = './data/ml-1m'
    data_dir = './data/ml-20m/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    top_k = 20
    multi_factor = 5

    # item context tensor
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
        item_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
        item_ppr_dict = pickle.load(f)

    pf = PPRfeatures(data_dir, top_k, item_idx_dict, item_ppr_dict)
    item_idx_tensor = pf.idx_tensor()
    item_scr_tensor = pf.score_tensor()
    del item_idx_dict
    del item_ppr_dict

    # user context tensor
    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
        user_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
        user_ppr_dict = pickle.load(f)

    pf = PPRfeatures(data_dir, top_k, user_idx_dict, user_ppr_dict)
    user_idx_tensor = pf.idx_tensor()
    user_scr_tensor = pf.score_tensor()
    del user_idx_dict
    del user_ppr_dict
