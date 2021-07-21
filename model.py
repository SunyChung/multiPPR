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

        # self.inter_layer = InterLin_1(input_dim=self.cat_dim, output_dim=1)

        # self.inter_layer = InterLin_3(input_dim=self.cat_dim,
        #                          hidden1=self.cat_dim // 2,
        #                          hidden2=self.cat_dim // 4,
        #                          output_dim=1)


        # self.inter_layer = InterLin_5(input_dim=self.cat_dim,
        #                               hidden1=self.cat_dim // 2,
        #                               hidden2=self.cat_dim // 4,
        #                               output_dim=1)

        # self.inter_layer = InterLin_5_norm(input_dim=self.cat_dim,
        #                                    hidden1=self.cat_dim // 2,
        #                                    hidden2=self.cat_dim // 4,
        #                                    output_dim=1)

        self.inter_layer = InterLin_5_no_relu(input_dim=self.cat_dim,
                                           hidden1=self.cat_dim // 2,
                                           hidden2=self.cat_dim // 4,
                                           output_dim=1)

        # self.inter_layer = InterLin_7(input_dim=self.cat_dim,
        #                               hidden1=self.cat_dim // 2,
        #                               hidden2=self.cat_dim // 4,
        #                               hidden3=self.cat_dim // 8,
        #                               output_dim=1)

        # self.inter_layer = InterLin_8(input_dim=self.cat_dim,
        #                               hidden1=self.cat_dim // 2,
        #                               hidden2=self.cat_dim // 4,
        #                               hidden3=self.cat_dim // 8,
        #                               output_dim=1)


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
        # result = torch.sigmoid(self.inter_layer(interaction_cat))
        # with BCEWithLogitsLoss()
        result = self.inter_layer(interaction_cat)
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


# epoch 20 에서, 중간에 loss 가 1 까지 올라갔으나, test 에서 0.24, 0.28 로 엄청 나쁘진 않음 ...
# epoch 을 늘려서 더 해 봐야 하나 ?!
# epoch 50 에서는 심지어 loss 가 3 이 넘어도, 0.31, 0.32 가 나옴 -_
# relu() 가 없어서 그런지 epoch 을 반복하면 loss 가 22 까지 올라감 ;; evaluation 값은 좋음 ...
# 뭔가 판단할 기준이 제대로 없는 ... BPR loss 함께 써야 될 듯 ??
class InterLin_5_no_relu(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(InterLin_5_no_relu, self).__init__()

        self.ln1 = nn.Linear(input_dim, hidden1)
        self.ln2 = nn.Linear(hidden1, hidden1)
        self.ln3 = nn.Linear(hidden1, hidden2)
        self.ln4 = nn.Linear(hidden2, hidden2)
        self.ln5 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.ln4(x)
        x = self.ln5(x)
        return x


class InterLin_5_norm(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(InterLin_5_norm, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.ln1 = nn.LayerNorm(hidden1, eps=1e-4)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2, eps=1e-4)
        self.fc3 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        h1 = self.ln1(F.relu(self.fc1(x)))  # 32
        h2 = self.ln2(F.relu(self.fc2(h1) + h1))
        h3 = self.ln3(F.relu(self.fc3(h2) + h1 + h2))
        return h3


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
