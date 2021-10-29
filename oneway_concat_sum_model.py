import torch
import torch.nn as nn
import torch.nn.functional as F


class oneway_concat(nn.Module):
    def __init__(self, item_idx_tensor, user_idx_tensor,
                 n_items, n_users, emb_dim, multi_factor, top_k):
        super(oneway_concat, self).__init__()

        self.item_idx_tensor = item_idx_tensor
        # torch.Size([batch_size, multi_factor x # of neighbors])
        self.user_idx_tensor = user_idx_tensor
        # torch.Size([batch_size, multi_factor x # of neighbors])

        self.multi_factor = multi_factor
        self.emb_dim = emb_dim

        self.user_emb = nn.Embedding(n_users, emb_dim).to('cuda')
        self.item_emb = nn.Embedding(n_items, emb_dim).to('cuda')
        print('\n using default weight initialization')
        print('\nuser_emb min : ', torch.min(self.user_emb.weight))
        print('user_emb max : ', torch.max(self.user_emb.weight))
        print('user_emb mean : ', torch.mean(self.user_emb.weight))
        print('user_emb std : ', torch.std(self.user_emb.weight))

        print('\nitem_emb min : ', torch.min(self.item_emb.weight))
        print('item_emb max : ', torch.max(self.item_emb.weight))
        print('item_emb mean : ', torch.mean(self.item_emb.weight))
        print('item_emb std : ', torch.std(self.item_emb.weight))

        # self.concat_net = concat_linear_1(input_dim=2,
        #                                           output_dim=1).to('cuda')
        self.concat_net = concat_linear_2(input_dim=2,
                                          output_dim=1).to('cuda')

    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs]
        # torch.Size([sample_size, 100])
        user_neigh_emb = self.user_emb(user_neighs)
        # torch.Size([sample_size, 100, 32])
        user_sum_emb = torch.sum(user_neigh_emb, 1, keepdim=True)
        # torch.Size([sample_size, 1, 32])

        item_neighs = self.item_idx_tensor[item_idxs]
        item_neigh_emb = self.item_emb(item_neighs)
        item_sum_emb = torch.sum(item_neigh_emb, 1, keepdim=True)

        user_item_emb = torch.cat((user_sum_emb, item_sum_emb), 1)
        # user_item_emb_1 : torch.Size([sample_size, 2, 32])
        concat = self.concat_net((user_item_emb).permute(0, 2, 1))
        # input : torch.Size([sample_size, 32, 2])
        # -> output :  torch.Size([sample_size, 32, 1])
        results = torch.sigmoid(torch.mean(concat, dim=1))
        # print('results : ', results.shape)
        # torch.Size([sample_size, 1])
        # print('squeezed shape : ', results.squeeze().shape)
        # torch.Size([sample_size])
        return results.squeeze()


class concat_linear_2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(concat_linear_2, self).__init__()

        self.ln1 = nn.Linear(input_dim, input_dim * 10)
        self.ln2 = nn.Linear(input_dim * 10, input_dim * 100)
        self.ln3 = nn.Linear(input_dim * 100, input_dim * 100)
        self.ln4 = nn.Linear(input_dim * 100, input_dim * 10)
        self.ln5 = nn.Linear(input_dim * 10, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = self.ln5(x)
        return x


class concat_linear_1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(concat_linear_1, self).__init__()

        self.ln = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.ln(x))
        return x
