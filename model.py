import torch
import torch.nn as nn


class ContextualizedNN(nn.Module):
    def __init__(self, item_idx_tensor, item_scr_tensor, user_idx_tensor, user_scr_tensor,
                         item_embedding, user_embedding):
        super(ContextualizedNN, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.item_idx_tensor = item_idx_tensor.to(self.device)
        self.item_scr_tensor = item_scr_tensor.to(self.device)
        self.user_idx_tensor = user_idx_tensor.to(self.device)
        self.user_scr_tensor = user_scr_tensor.to(self.device)
        self.item_emb = item_embedding.to(self.device)
        self.user_emb = user_embedding.to(self.device)

        self.inter_input_dim = self.item_emb.embedding_dim
        # print('inter_input_dim : ', self.inter_input_dim)
        self.inter_lin = InterLin(input_dim=self.inter_input_dim,
                                  hidden=self.inter_input_dim//10,
                                  output_dim=1)

    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs].to(self.device)
        # print('user_neighs shape : ', user_neighs.shape)  # [batch_size, 100=(5x20)]
        neigh_emb = self.user_emb(user_neighs).to(self.device)
        # print('neigh_emb shape : ', neigh_emb.shape)  # [batch_size, 100(=5x20), emb_dim]
        neigh_score = self.user_scr_tensor[user_neighs].to(self.device)
        # print('neigh_score shape : ', neigh_score.shape)  # [batch_size, 100(=5x20), 100(=5x20)]
        scored_user_emb = torch.matmul(neigh_score, neigh_emb)
        # print('scored_user_emb shape : ', scored_user_emb.shape)  # torch.Size([500, 100, emb_dim])

        item_neighs = self.item_idx_tensor[item_idxs].to(self.device)
        item_neigh_emb = self.item_emb(item_neighs).to(self.device)
        item_neigh_scr = self.item_scr_tensor[item_neighs].to(self.device)
        scored_item_emb = torch.matmul(item_neigh_scr, item_neigh_emb)

        interaction = scored_user_emb * scored_item_emb  # torch.Size([500, 100, emb_dim])
        # print('interaction shape : ', interaction.shape)
        result = torch.sigmoid(self.inter_lin(interaction))
        return torch.mean(result, dim=1).squeeze()


class InterLin(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(InterLin, self).__init__()

        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim

        self.ln1 = nn.Linear(input_dim, hidden)
        self.ln2 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        return x
