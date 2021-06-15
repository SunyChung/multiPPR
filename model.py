import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualizedNN(nn.Module):
    def __init__(self, item_idx_tensor, item_scr_tensor, user_idx_tensor, user_scr_tensor,
                         item_embedding, user_embedding):
        super(ContextualizedNN, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.item_idx_tensor = item_idx_tensor.to(self.device)
        self.item_scr_tensor = item_scr_tensor.to(self.device)
        self.user_idx_tensor = user_idx_tensor.to(self.device)
        self.user_scr_tensor = user_scr_tensor.to(self.device)
        # emb_dim = 100
        self.item_emb = item_embedding.to(self.device)  # nn.Embedding(n_items, emb_dim)
        self.user_emb = user_embedding.to(self.device)  # nn.Embedding(len(unique_uidx), emb_dim)

        self.inter_input_dim = self.item_emb.embedding_dim
        self.inter_lin = InterLin(input_dim=self.inter_input_dim,
                                  hidden=self.inter_input_dim//10,
                                  output_dim=1)

    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs].to(self.device)
        neigh_emb = self.user_emb(user_neighs).to(self.device)
        neigh_score = self.user_scr_tensor[user_neighs].to(self.device)
        scored_user_emb = torch.matmul(neigh_score, neigh_emb)

        item_neighs = self.item_idx_tensor[item_idxs].to(self.device)
        item_neigh_emb = self.item_emb(item_neighs).to(self.device)
        item_neigh_scr = self.item_scr_tensor[item_neighs].to(self.device)
        scored_item_emb = torch.matmul(item_neigh_scr, item_neigh_emb)

        interaction = scored_user_emb * scored_item_emb
        result = torch.sigmoid(self.inter_lin(interaction))
        # print('result shape : ', result.shape)
        # for batch training : torch.Size([500, 100, 1])
        # for one id evaluation : torch.Size([100, 1])
        return torch.mean(result, dim=-2).squeeze()


class InterLin(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(InterLin, self).__init__()

        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim

        self.ln1 = nn.Linear(input_dim, hidden)
        self.ln2 = nn.Linear(hidden, output_dim)
        self.ln3 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return x
