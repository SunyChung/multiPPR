import pickle
import os
import torch
import torch.nn as nn

from layers import LinearRep, PPRfeatures


class ContextualizedNN(nn.Module):
    def __init__(self, item_idx_emb, item_scr_emb, user_idx_emb, user_scr_emb,
                 multi_factor, input_dim, hidden_dim, output_dim, final_dim):
        super(ContextualizedNN, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.item_idx_emb = item_idx_emb.to(self.device)
        self.item_scr_emb = item_scr_emb.to(self.device)
        self.user_idx_emb = user_idx_emb.to(self.device)
        self.user_scr_emb = user_scr_emb.to(self.device)

        self.inter_input_dim = output_dim // multi_factor
        self.item_rep = LinearRep(input_dim, hidden_dim, output_dim, multi_factor, item_idx_emb, item_scr_emb).to(self.device)
        self.user_rep = LinearRep(input_dim, hidden_dim, output_dim, multi_factor, user_idx_emb, user_scr_emb).to(self.device)
        self.inter_lin = nn.Linear(self.inter_input_dim, final_dim).to(self.device)

    def forward(self, user_idxs, item_idxs):
        user_rep = self.user_rep(torch.LongTensor(user_idxs).to(self.device))
        # print('user_rep shape : ', user_rep.shape)  # torch.Size([batch_size, 5, 10])
        item_rep = self.item_rep(torch.LongTensor(item_idxs).to(self.device))
        # print('item_rep shape : ', item_rep.shape)  # torch.Size([batch_size, 5, 10])
        interaction = item_rep * user_rep
        # print('interaction shape : ', interaction.shape)  # torch.Size([batch_size, 5, 10])
        result = torch.sigmoid(self.inter_lin(interaction))
        # print('result shape : ', result.shape)  # torch.Size([batch_size, 5, 1])
        # 근데, 이 형태로는 target 이랑 사이즈가 안 맞음 !
        # [batch_size, factor_size=5, final_dim=1]
        # loss 계산할 때, 조절하면 됨 ! DONE
        return result


if __name__ == '__main__':
    data_dir = './data/ml-1m'
    top_k = 20
    multi_factor = 5

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

    CN = ContextualizedNN(item_idx_emb, item_scr_emb, user_idx_emb, user_scr_emb,
                          multi_factor,
                          input_dim=top_k * multi_factor,
                          hidden_dim=top_k * multi_factor * 5,
                          output_dim=multi_factor * 10,
                          final_dim=1)
    result = CN([1, 4, 5, 8, 9, 10], [1, 1, 1, 1, 1, 1])
    print('results = ', result)
    print('results shape : ', result.shape)  # torch.Size([batch_size, 5, 1])
    # 이걸 target 이랑 맞추려면, factor 축으로 평균낸 값 써야 함 !
    print('results factor mean : ', torch.mean(result, dim=1))
    print('reshaped results : ', torch.mean(result, dim=1).shape)  # torch.Size([batch_size, 1])

    print('results factor mean : ', torch.mean(result, dim=2))
    print('reshaped results : ', torch.mean(result, dim=2).shape)  # torch.Size([batch_size, 5])
