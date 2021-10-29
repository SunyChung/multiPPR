import torch
import torch.nn as nn
import torch.nn.functional as F


class oneway_concat(nn.Module):
    def __init__(self, item_idx_tensor, user_idx_tensor,
                 n_items, n_users, emb_dim, multi_factor, top_k):
        super(oneway_concat, self).__init__()

        self.item_idx_tensor = item_idx_tensor
        # torch.Size([3515, multi_factor x # of neighbors])
        self.user_idx_tensor = user_idx_tensor
        # torch.Size([6034, multi_factor x # of neighbors])

        self.multi_factor = multi_factor
        self.emb_dim = emb_dim

        self.user_emb = nn.Embedding(n_users, emb_dim).to('cuda')
        self.item_emb = nn.Embedding(n_items, emb_dim).to('cuda')
        print('\n using default weight initialization')

        # print('\n kaiming_normal_ weight initialization')
        # nn.init.kaiming_normal_(self.user_emb.weight)
        # nn.init.kaiming_normal_(self.item_emb.weight)

        # print('\nusing xavier_normal_ weight initialization')
        # nn.init.xavier_normal_(self.user_emb.weight)
        # nn.init.xavier_normal_(self.item_emb.weight)
        print('\nuser_emb min : ', torch.min(self.user_emb.weight))
        print('user_emb max : ', torch.max(self.user_emb.weight))
        print('user_emb mean : ', torch.mean(self.user_emb.weight))
        print('user_emb std : ', torch.std(self.user_emb.weight))

        print('\nitem_emb min : ', torch.min(self.item_emb.weight))
        print('item_emb max : ', torch.max(self.item_emb.weight))
        print('item_emb mean : ', torch.mean(self.item_emb.weight))
        print('item_emb std : ', torch.std(self.item_emb.weight))

        self.concat_net_1 = concat_linear(input_dim=multi_factor * top_k * 2,
                                          output_dim=1).to('cuda')
        # self.concat_net_2 = concat_linear(input_dim=emb_dim * 2,
        #                                   output_dim=1).to('cuda')

    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs]
        # torch.Size([3515, 100 = (multi_factor x # of neighbors)])
        user_neigh_emb = self.user_emb(user_neighs)
        # torch.Size([3515, 100, 64])

        item_neighs = self.item_idx_tensor[item_idxs]
        item_neigh_emb = self.item_emb(item_neighs)

        user_item_emb_1 = torch.cat((user_neigh_emb, item_neigh_emb), 1)
        # torch.Size([3515, 200, 64])
        concat_1 = self.concat_net_1((user_item_emb_1).permute(0, 2, 1))
        # torch.Size([3515, 64, 1])
        results = torch.sigmoid(torch.mean(concat_1, dim=1))
        # print('results : ', results.shape)
        # torch.Size([3515, 1])
        return results.squeeze()


class concat_linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(concat_linear, self).__init__()

        self.ln1 = nn.Linear(input_dim, input_dim//2)
        self.ln2 = nn.Linear(input_dim//2, input_dim//4)
        self.ln3 = nn.Linear(input_dim//4, input_dim//8)
        self.ln4 = nn.Linear(input_dim//8, input_dim//16)
        self.ln5 = nn.Linear(input_dim//16, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = self.ln5(x)
        return x
