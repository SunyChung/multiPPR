import numpy as np
import torch
import torch.nn as nn


class LinearRep(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, id_emb, scr_emb):
        super(LinearRep, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.id_emb = id_emb
        self.scr_emb = scr_emb

        self.ln1 = nn.Linear(input_dim, input_dim)
        self.ln2 = nn.Linear(input_dim, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, index):
        x = self.ln1(self.id_emb(torch.LongTensor(index))).to(self.device)
        x = torch.mul(x, self.scr_emb(torch.LongTensor(index))).to(self.device)
        x = self.ln2(x)
        x = self.ln3(x)
        return x


class PPRfeatures(object):
    def __init__(self, data_dir, top_k, idx_dict, ppr_dict):
        self.data_dir = data_dir
        self.top_k = top_k
        self.idx_values = np.array(list(idx_dict.values()))  # shape: (3533, 5, 3532)
        self.ppr_scores = np.array(list(ppr_dict.values())) # shape: (3533, 5, 3532)

    def idx_embeds(self):
        idx_top_k = self.idx_values[:, :, :self.top_k]
        idx_emb = nn.Embedding(idx_top_k.shape[0], idx_top_k.shape[1]*idx_top_k.shape[2])
        print('idx_emb shape : ', idx_emb)  # Embedding(3533, 100) when top_k = 20
        idx_emb.weight = nn.Parameter(torch.Tensor(idx_top_k.reshape(idx_top_k.shape[0], -1)))
        print('idx_emb.weight : ', idx_emb.weight)
        return idx_emb

    def score_embeds(self):
        score_top_k = self.ppr_scores[:, :, :self.top_k]
        score_emb = nn.Embedding(score_top_k.shape[0], score_top_k.shape[1]*score_top_k.shape[2])
        print('score_emb shape : ', score_emb)
        score_emb.weight = nn.Parameter(torch.Tensor(score_top_k.reshape(score_top_k.shape[0], -1)))
        print('score_emb.weight : ', score_emb.weight)  # idx_emb.weight.shape : torch.Size([3533, 100])
        return score_emb
