import os
import pickle
import numpy as np
import torch
import torch.nn as nn


class LinearRep(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, multi_factor, id_emb, scr_emb):
        super(LinearRep, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.multi_factor = multi_factor
        self.reshaped_dim = output_dim // multi_factor
        self.id_emb = id_emb.to(self.device)
        self.scr_emb = scr_emb.to(self.device)

        self.ln1 = nn.Linear(input_dim, input_dim).to(self.device)
        self.ln2 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.ln3 = nn.Linear(hidden_dim, output_dim).to(self.device)

    def forward(self, index):
        x = self.ln1(self.id_emb(index))
        x = torch.mul(x, self.scr_emb(index))
        x = self.ln2(x)
        x = self.ln3(x).reshape(-1, self.multi_factor, self.reshaped_dim)
        return x


class PPRfeatures(object):
    def __init__(self, data_dir, top_k, idx_dict, ppr_dict):
        self.data_dir = data_dir
        self.top_k = top_k
        self.idx_values = np.array(list(idx_dict.values()))
        self.ppr_scores = np.array(list(ppr_dict.values()))

    def idx_tensor(self):
        # dictionary shape : [all nodes idxs , multi-factors, ranked idxs]
        idx_top_k = self.idx_values[:, :, 1:self.top_k+1]
        idx_tensor = torch.LongTensor(idx_top_k.reshape(idx_top_k.shape[0], -1))
        print('idx_tensor shape : ', idx_tensor.shape)  # torch.Size([3505, 100])
        print('idx_tensor type : ', idx_tensor.type)
        return idx_tensor

    def score_tensor(self):
        # dictionary shape : [all nodes idxs, multi-factors, ranked scores]
        score_top_k = self.ppr_scores[:, :, 1:self.top_k+1]
        score_tensor = torch.FloatTensor(score_top_k.reshape(score_top_k.shape[0], -1))
        print('score_tensor shape : ', score_tensor.shape)
        print('score_tensor type : ', score_tensor.type)
        return score_tensor

    # previous try with nn.Embedding
    # idx_embeds got torch.FloatTensor(Tensor) as the dtype,
    # thus, can not be used as the another input to the nn.Embedding
    def idx_embeds(self):
        idx_top_k = self.idx_values[:, :, 1:self.top_k+1]
        idx_emb = nn.Embedding.from_pretrained(
            torch.Tensor(idx_top_k.reshape(idx_top_k.shape[0], -1)), freeze=True)
        # 근데 인덱스의 정수 값만 사용하기 위해
        # (여기에서 반환받는 idx 값이 다시 embedding 의 index 로 사용되기 때문에 torch.LongTensor 만 사용 가능 !)
        # type 을 바꾸고 나면, embedding 이 아닌가?! YES 그냥 type 지정한 tensor 가 되어 버림 .... ;
        # 그냥 embedding 이 아니라, tensor 로 만들어서 호출해도 되는데, 이러면 또 index 는 tensor 가 아니라 숫자로 써야 함 ....
        # longed_idx_emb = idx_emb.weight.type(torch.LongTensor)
        # tensor 에서 multiple index 의 값을 동시에 반환 받으려면, 이중으로 리스트 값만 넘기면 됨
        # ex. idx_tensor[[0, 4, 7, 8]]
        # 어차피 gradient 안 구하면, 이건 그냥 tensor 써야 됨.. 다시 수정
        return idx_emb

    def score_embeds(self):
        score_top_k = self.ppr_scores[:, :, 1:self.top_k+1]
        score_emb = nn.Embedding.from_pretrained(
            torch.Tensor(score_top_k.reshape(score_top_k.shape[0], -1)), freeze=True)
        return score_emb


if __name__ == '__main__':
    data_dir = './data/ml-1m'
    top_k = 20
    multi_factor = 5

    # item context embedding
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
        item_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
        item_ppr_dict = pickle.load(f)

    pf = PPRfeatures(data_dir, top_k, item_idx_dict, item_ppr_dict)
    item_idx_tensor = pf.idx_tensor()
    item_scr_tensor = pf.score_tensor()
    del item_idx_dict
    del item_ppr_dict

    # user context embedding
    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
        user_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
        user_ppr_dict = pickle.load(f)

    pf = PPRfeatures(data_dir, top_k, user_idx_dict, user_ppr_dict)
    user_idx_tensor = pf.idx_tensor()
    user_scr_tensor = pf.score_tensor()
    del user_idx_dict
    del user_ppr_dict
