import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualizedNN(nn.Module):
    def __init__(self, item_idx_tensor, item_scr_tensor, user_idx_tensor, user_scr_tensor,
                         item_embedding, user_embedding, multi_factor, top_k):
        super(ContextualizedNN, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.item_idx_tensor = item_idx_tensor.to(self.device)
        self.item_scr_tensor = item_scr_tensor.to(self.device)
        self.user_idx_tensor = user_idx_tensor.to(self.device)
        self.user_scr_tensor = user_scr_tensor.to(self.device)

        self.item_emb = item_embedding.to(self.device)  # nn.Embedding(n_items, emb_dim)
        self.user_emb = user_embedding.to(self.device)  # nn.Embedding(len(unique_uidx), emb_dim)

        self.cat_dim = self.item_emb.embedding_dim * 2
        self.inter_lin = InterLin(input_dim=self.cat_dim,
                                  hidden1=self.cat_dim // 2, hidden2=self.cat_dim // 4,
                                  output_dim=1)

    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs].to(self.device)
        # [top_k x multi_factor]
        neigh_emb = self.user_emb(user_neighs).to(self.device)
        # [top_k x multi_factor] x embed_dim
        neigh_score = self.user_scr_tensor[user_neighs].to(self.device)
        scored_user_emb = torch.matmul(neigh_score, neigh_emb)
        # batch_size x [top_k x multi_factor] x embed_dim

        item_neighs = self.item_idx_tensor[item_idxs].to(self.device)
        item_neigh_emb = self.item_emb(item_neighs).to(self.device)
        item_neigh_scr = self.item_scr_tensor[item_neighs].to(self.device)
        scored_item_emb = torch.matmul(item_neigh_scr, item_neigh_emb)

        interaction_cat = torch.cat((scored_user_emb, scored_item_emb), dim=-1)
        # print('interaction_cat shape : ', interaction_cat.shape)
        # batch_size, [multi_factor x top_k], embedding_dim x 2
        result = torch.sigmoid(self.inter_lin(interaction_cat))
        return torch.mean(result, dim=-2).squeeze()


class InterLin(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(InterLin, self).__init__()
        # batch_size, [multi_factor x top_k], embedding_dim x 2
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_dim = output_dim

        self.ln1 = nn.Linear(input_dim, hidden1)
        self.ln2 = nn.Linear(hidden1, hidden2)
        self.ln3 = nn.Linear(hidden2, output_dim)
        # batch_size, [multi_factor x top_k], output_dim

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return x
