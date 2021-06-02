import numpy as np
import pickle
import os
import torch
import torch.nn as nn

from layers import LinearRep, PPRfeatures


class ContextualizedNN(nn.Module):
    def __init__(self, data_dir, item_idx_emb, item_scr_emb, user_idx_emb, user_scr_emb,
                 input_dim, hidden_dim, output_dim, final_dim):
        super(ContextualizedNN, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.item_idx_emb = item_idx_emb
        self.item_scr_emb = item_scr_emb
        self.user_idx_emb = user_idx_emb
        self.user_scr_emb = user_scr_emb

        self.item_rep = LinearRep(input_dim, hidden_dim, output_dim, item_idx_emb, item_scr_emb)
        self.user_rep = LinearRep(input_dim, hidden_dim, output_dim, user_idx_emb, user_scr_emb)
        self.inter_lin = nn.Linear(output_dim, final_dim)

    def forward(self, item_idxs, user_idxs):
        item_rep = self.item_rep(item_idxs)
        user_rep = self.user_rep(user_idxs)
        interaction = item_rep * user_rep
        result = torch.sigmoid(self.inter_lin(interaction))
        return result


if __name__ == '__main__':
    data_dir = './data/ml-1m'
    top_k = 20

    # item context dictionary making
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
        item_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
        item_ppr_dict = pickle.load(f)

    pf = PPRfeatures(data_dir, top_k, item_idx_dict, item_ppr_dict)
    item_idx_emb = pf.idx_embeds()
    item_scr_emb = pf.score_embeds()
    del item_idx_dict
    del item_ppr_dict

    # user context dictionary making
    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
        user_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
        user_ppr_dict = pickle.load(f)

    pf = PPRfeatures(data_dir, top_k, user_idx_dict, user_ppr_dict)
    user_idx_emb = pf.idx_embeds()
    user_scr_emb = pf.score_embeds()
    del user_idx_dict
    del user_ppr_dict

    # input, hidden, output dimension 잘 생각할 것 ...
    CN = ContextualizedNN(data_dir, item_idx_emb, item_scr_emb, user_idx_emb, user_scr_emb,
                          input_dim=top_k*5, hidden_dim=top_k*5*5, output_dim=top_k, final_dim=1)
    result = CN([1, 4, 5, 8, 9], [1, 1, 1, 1, 1])
    print('results = ', result)
