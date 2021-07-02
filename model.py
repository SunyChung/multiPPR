import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualizedNN(nn.Module):
    def __init__(self, item_idx_tensor, item_scr_tensor,
                 user_idx_tensor, user_scr_tensor,
                 item_embedding, user_embedding):
        super(ContextualizedNN, self).__init__()

        self.item_idx_tensor = item_idx_tensor  # torch.Size([3516, 250])
        self.item_scr_tensor = item_scr_tensor
        self.user_idx_tensor = user_idx_tensor  # torch.Size([6031, 250])
        self.user_scr_tensor = user_scr_tensor

        self.item_emb = item_embedding
        self.user_emb = user_embedding

        self.cat_dim = self.item_emb.embedding_dim * 2
        self.inter_lin = InterLin(input_dim=self.cat_dim,
                                  hidden1=self.cat_dim // 2, hidden2=self.cat_dim // 4,
                                  output_dim=1)

    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs]
        # print('user_neighs shape : ', user_neighs.shape)  # torch.Size([3516, 250])
        neigh_emb = self.user_emb(user_neighs)
        # print('neigh_emb shape : ', neigh_emb.shape)  # torch.Size([3516, 250, 100])
        neigh_score = self.user_scr_tensor[user_neighs]
        # print('neigh_score shape : ', neigh_score.shape)  # torch.Size([3516, 250, 250])
        scored_user_emb = torch.matmul(neigh_score, neigh_emb)
        # print('scored_user_emb shape : ', scored_user_emb.shape)  # torch.Size([3516, 250, 100])

        item_neighs = self.item_idx_tensor[item_idxs]
        # print('item_neighs shape : ', item_neighs.shape)  # torch.Size([3516, 250])
        item_neigh_emb = self.item_emb(item_neighs)
        # print('item_neigh_emb shape : ', item_neigh_emb.shape)  # torch.Size([3516, 250, 100])
        item_neigh_scr = self.item_scr_tensor[item_neighs]
        # print('item_neigh_scr shape : ', item_neigh_scr.shape)  # torch.Size([3516, 250, 250])
        scored_item_emb = torch.matmul(item_neigh_scr, item_neigh_emb)
        # print('scored_item_emb shape : ', scored_item_emb.shape)  # torch.Size([3516, 250, 100])

        # 여기서 에러 발생하는데, 대체 월요일부터 뭐가 문제인지 몰 것다 -_;;
        # .to(self.device) 로 GPU 로 보낸 데이터랑, CPU 에 있는 데이터랑 따로 놀아서 생기는 문제;
        interaction_cat = torch.cat((scored_user_emb, scored_item_emb), 2)
        # print('interaction_cat : ', interaction_cat)
        result = torch.sigmoid(self.inter_lin(interaction_cat))
        return torch.mean(result, dim=-2).squeeze()


class InterLin(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(InterLin, self).__init__()

        self.ln1 = nn.Linear(input_dim, hidden1)
        self.ln2 = nn.Linear(hidden1, hidden2)
        self.ln3 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return x
