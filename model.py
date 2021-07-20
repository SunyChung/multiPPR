import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualizedNN(nn.Module):
    def __init__(self, item_idx_tensor, item_scr_tensor,
                 user_idx_tensor, user_scr_tensor,
                 item_embedding, user_embedding):
        super(ContextualizedNN, self).__init__()

        self.item_idx_tensor = item_idx_tensor  # torch.Size([3515, multi_factor x # of neighbors])
        self.item_scr_tensor = item_scr_tensor
        self.user_idx_tensor = user_idx_tensor  # torch.Size([6034, multi_factor x # of neighbors])
        self.user_scr_tensor = user_scr_tensor

        print('\ndefault weight initialization')
        # print('\nusing xavier_normal_ weight initialization')
        # print(\n kaiming_normal_ weight initialization')

        self.item_emb = item_embedding
        # print('item_emb : ', self.item_emb.weight)
        print('item_emb min : ', torch.min(self.item_emb.weight))
        print('item_emb max : ', torch.max(self.item_emb.weight))
        # nn.init.xavier_normal_(self.item_emb.weight)
        # print('item_emb initialized : ', self.item_emb.weight)
        # nn.init.kaiming_normal_(self.item_emb.weight)

        self.user_emb = user_embedding
        # print('user_emb : ', self.user_emb.weight)
        print('user_emb min : ', torch.min(self.user_emb.weight))
        print('user_emb max : ', torch.max(self.user_emb.weight))
        # nn.init.xavier_normal_(self.user_emb.weight)
        # print('user_emb initialized : ', self.user_emb.weight)
        # nn.init.kaiming_normal_(self.user_emb.weight)


        self.cat_dim = self.item_emb.embedding_dim * 2

        # self.inter_lin_1 = InterLin_1(input_dim=self.cat_dim, output_dim=1)

        # self.inter_lin_3 = InterLin_3(input_dim=self.cat_dim,
        #                          hidden1=self.cat_dim // 2,
        #                          hidden2=self.cat_dim // 4,
        #                          output_dim=1)


        # self.inter_lin_5 = InterLin_5(input_dim=self.cat_dim,
        #                               hidden1=self.cat_dim // 2,
        #                               hidden2=self.cat_dim // 4,
        #                               output_dim=1)

        # self.inter_lin_5_norm = InterLin_5_norm(input_dim=self.cat_dim,
        #                                       hidden1=self.cat_dim // 2,
        #                                       hidden2=self.cat_dim // 4,
        #                                       output_dim=1)

        # self.inter_lin_7 = InterLin_7(input_dim=self.cat_dim,
        #                               hidden1=self.cat_dim // 2,
        #                               hidden2=self.cat_dim // 4,
        #                               hidden3=self.cat_dim // 8,
        #                               output_dim=1)

        self.inter_lin_8 = InterLin_8(input_dim=self.cat_dim,
                                      hidden1=self.cat_dim // 2,
                                      hidden2=self.cat_dim // 4,
                                      hidden3=self.cat_dim // 8,
                                      output_dim=1)


    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs]
        # torch.Size([3515(=batch_size=# of items), multi_factor x # of neighbors])
        neigh_emb = self.user_emb(user_neighs)
        # torch.Size([3515(=batch_size), multi_factor x # of neighbors, embedding_dim])
        neigh_score = self.user_scr_tensor[user_neighs]
        # torch.Size([3515(=batch_size), multi_factor x # of neighbors, multi_factor x # of neighbors])
        scored_user_emb = torch.matmul(neigh_score, neigh_emb)
        # torch.Size([3515(=batch_size), multi_factor x # of neighbors, embedding_dim])

        item_neighs = self.item_idx_tensor[item_idxs]
        # torch.Size([3515(=batch_size=# of items), multi_factor x # of neighbors])
        item_neigh_emb = self.item_emb(item_neighs)
        # torch.Size([3515(=batch_size), multi_factor x # of neighbors, embedding_dim])
        item_neigh_scr = self.item_scr_tensor[item_neighs]
        # torch.Size([3515(=batch_size), multi_factor x # of neighbors, multi_factor x # of neighbors])
        scored_item_emb = torch.matmul(item_neigh_scr, item_neigh_emb)
        # torch.Size([3515(=batch_size), multi_factor x # of neighbors, embedding_dim])

        interaction_cat = torch.cat((scored_user_emb, scored_item_emb), 2)
        # with BCELoss()
        # result = torch.sigmoid(self.inter_lin_1(interaction_cat))
        # result = torch.sigmoid(self.inter_lin_3(interaction_cat))
        # with BCEWithLogitsLoss()
        # result = self.inter_lin_1(interaction_cat)
        # result = self.inter_lin_5(interaction_cat)
        # result = self.inter_lin_7(interaction_cat)
        result = self.inter_lin_8(interaction_cat)
        return torch.mean(result, dim=-2).squeeze()


class InterLin_1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InterLin_1, self).__init__()

        self.ln = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x = F.relu(self.ln(x))
        x = self.ln(x)
        return x


class InterLin_3(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(InterLin_3, self).__init__()

        self.ln1 = nn.Linear(input_dim, hidden1)
        self.ln2 = nn.Linear(hidden1, hidden2)
        self.ln3 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return x


class InterLin_5(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(InterLin_5, self).__init__()

        self.ln1 = nn.Linear(input_dim, hidden1)
        self.ln2 = nn.Linear(hidden1, hidden1)
        self.ln3 = nn.Linear(hidden1, hidden2)
        self.ln4 = nn.Linear(hidden2, hidden2)
        self.ln5 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = self.ln5(x)
        return x


class InterLin_5_norm(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(InterLin_5_norm, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.ln1 = nn.LayerNorm(hidden1, eps=1e-4)
        self.fc2 = nn.Linear(hidden1, hidden1)
        self.ln2 = nn.LayerNorm(hidden1, eps=1e-4)
        self.fc3 = nn.Linear(hidden1, hidden2)
        self.ln3 = nn.LayerNorm(hidden2, eps=1e-4)
        self.fc4 = nn.Linear(hidden2, hidden2)
        self.ln4 = nn.LayerNorm(hidden2, eps=1e-4)
        self.fc5 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        h1 = self.ln1(F.relu(self.fc1(x)))
        h2 = self.ln2(F.relu(self.fc2(h1) + h1))
        h3 = self.ln3(F.relu(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(F.relu(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.fc5(h4)
        return h5


class InterLin_7(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3, output_dim):
        super(InterLin_7, self).__init__()

        self.ln1 = nn.Linear(input_dim, input_dim)
        self.ln2 = nn.Linear(input_dim, hidden1)
        self.ln3 = nn.Linear(hidden1, hidden2)
        self.ln4 = nn.Linear(hidden2, hidden2)
        self.ln5 = nn.Linear(hidden2, hidden3)
        self.ln6 = nn.Linear(hidden3, hidden3)
        self.ln7 = nn.Linear(hidden3, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = F.relu(self.ln5(x))
        x = F.relu(self.ln6(x))
        x = self.ln7(x)
        return x


class InterLin_8(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3, output_dim):
        super(InterLin_8, self).__init__()

        self.ln1 = nn.Linear(input_dim, input_dim)
        self.ln2 = nn.Linear(input_dim, hidden1)
        self.ln3 = nn.Linear(hidden1, hidden1)
        self.ln4 = nn.Linear(hidden1, hidden2)
        self.ln5 = nn.Linear(hidden2, hidden2)
        self.ln6 = nn.Linear(hidden2, hidden3)
        self.ln7 = nn.Linear(hidden3, hidden3)
        self.ln8 = nn.Linear(hidden3, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = F.relu(self.ln5(x))
        x = F.relu(self.ln6(x))
        x = F.relu(self.ln7(x))
        x = self.ln8(x)
        return x


'''
class InterLin_cnn(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(InterLin_cnn, self).__init__()

        self.
'''
