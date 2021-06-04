import numpy as np
import torch
import torch.nn as nn


class LinearRep(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, multi_factor, id_emb, scr_emb):
        super(LinearRep, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.multi_factor = multi_factor
        self.reshaped_dim = output_dim // multi_factor
        # print('reshaped_dim : ', self.reshaped_dim)  # 10
        self.id_emb = id_emb.to(self.device)
        self.scr_emb = scr_emb.to(self.device)

        self.ln1 = nn.Linear(input_dim, input_dim)
        self.ln2 = nn.Linear(input_dim, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, index):
        x = self.ln1(self.id_emb(torch.LongTensor(index)).to(self.device))
        x = torch.mul(x, self.scr_emb(torch.LongTensor(index)).to(self.device))
        x = self.ln2(x)
        x = self.ln3(x).reshape(-1, self.multi_factor, self.reshaped_dim)
        # print('x shape : ', x.shape)  # torch.Size([6, 5, 10])
        return x


class PPRfeatures(object):
    def __init__(self, data_dir, top_k, idx_dict, ppr_dict):
        self.data_dir = data_dir
        self.top_k = top_k
        self.idx_values = np.array(list(idx_dict.values()))
        self.ppr_scores = np.array(list(ppr_dict.values()))

    def idx_embeds(self):
        # multi-PPR doesn't filtered the first PPR node
        # check before remove it from the list !
        # SHOULD the 1st idx be the target id itself,
        # check before move on to the next step !
        # idx_top_k = self.idx_values[:, :, 1:self.top_k]
        idx_top_k = self.idx_values[:, :, :self.top_k]
        idx_emb = nn.Embedding(idx_top_k.shape[0], idx_top_k.shape[1]*idx_top_k.shape[2])
        print('idx_emb shape : ', idx_emb)
        idx_emb.weight = nn.Parameter(torch.Tensor(idx_top_k.reshape(idx_top_k.shape[0], -1)))
        # print('idx_emb.weight : ', idx_emb.weight)
        return idx_emb

    def score_embeds(self):
        # multi-PPR doesn't filtered the first PPR node
        # check before remove it from the list !
        # score_top_k = self.ppr_scores[:, :, 1:self.top_k]
        score_top_k = self.ppr_scores[:, :, :self.top_k]
        score_emb = nn.Embedding(score_top_k.shape[0], score_top_k.shape[1]*score_top_k.shape[2])
        print('score_emb shape : ', score_emb)
        score_emb.weight = nn.Parameter(torch.Tensor(score_top_k.reshape(score_top_k.shape[0], -1)))
        # print('score_emb.weight : ', score_emb.weight)
        return score_emb
