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

        # self.concat_net = concat_linear_5(input_dim=multi_factor * top_k * 2,
        #                                   output_dim=1).to('cuda')

        self.concat_net = concat_linear_8(input_dim=multi_factor * top_k * 2,
                                                  output_dim=1).to('cuda')

        # self.concat_net = concat_linear_15(input_dim=multi_factor * top_k * 2,
        #                                           output_dim=1).to('cuda')


    def forward(self, user_idxs, item_idxs):
        user_neighs = self.user_idx_tensor[user_idxs]
        # torch.Size([sample_size, 5 x 20 = (multi_factor x # of neighbors)])
        user_neigh_emb = self.user_emb(user_neighs)
        # torch.Size([sample_size, 100, 32])

        item_neighs = self.item_idx_tensor[item_idxs]
        item_neigh_emb = self.item_emb(item_neighs)

        user_item_emb = torch.cat((user_neigh_emb, item_neigh_emb), 1)
        # user_item_emb : torch.Size([batch_size, 100 x 2, 32])
        concat_out = self.concat_net((user_item_emb).permute(0, 2, 1))
        # torch.Size([batch_size, 32, 1])
        # if concat_linear_8 or concat_linear_15
        # torch.Size([batch_size, 32, 200])

        # results = torch.sigmoid(torch.mean(concat_out, dim=1))
        # for concat_linear_8
        # 평균 낼 때, dimension 주의해서 다시 볼 것 !!
        results = torch.sigmoid(torch.mean(concat_out, dim=2))
        # print('results : ', results.shape)
        # torch.Size([batch_size, 1])
        return results.squeeze()


class concat_linear_5(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(concat_linear_5, self).__init__()
        # 200 -> 100 -> 50 -> 25 -> 12 -> 1
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


class concat_linear_8(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(concat_linear_8, self).__init__()
        # 200 -> 100 -> 50 -> 25 -> 12 -> 25 -> 50 -> 100 -> 200
        self.ln1 = nn.Linear(input_dim, input_dim//2)
        self.ln2 = nn.Linear(input_dim//2, input_dim//4)
        self.ln3 = nn.Linear(input_dim//4, input_dim//8)
        self.ln4 = nn.Linear(input_dim//8, input_dim//16)
        self.ln5 = nn.Linear(input_dim//16, input_dim//8)
        self.ln6 = nn.Linear(input_dim//8, input_dim//4)
        self.ln7 = nn.Linear(input_dim//4, input_dim//2)
        self.ln8 = nn.Linear(input_dim//2, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = F.relu(self.ln5(x))
        x = F.relu(self.ln6(x))
        x = F.relu(self.ln7(x))
        x = self.ln9(x)
        return x


class concat_linear_15(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(concat_linear_15, self).__init__()
        # 200 -> 100 -> 100 -> 50 -> 50 -> 25 -> 25 -> 12 -> 12
        # -> 25 -> 25 -> 50 -> 50 -> 100 -> 100 -> 200
        self.ln1 = nn.Linear(input_dim, input_dim//2)
        self.ln2 = nn.Linear(input_dim//2, input_dim//2)
        self.ln3 = nn.Linear(input_dim//2, input_dim//4)
        self.ln4 = nn.Linear(input_dim//4, input_dim//4)
        self.ln5 = nn.Linear(input_dim//4, input_dim//8)
        self.ln6 = nn.Linear(input_dim//8, input_dim//8)
        self.ln7 = nn.Linear(input_dim//8, input_dim//16)
        self.ln8 = nn.Linear(input_dim//16, input_dim//16)
        self.ln9 = nn.Linear(input_dim//16, input_dim//8)
        self.ln10 = nn.Linear(input_dim//8, input_dim//8)
        self.ln11 = nn.Linear(input_dim//8, input_dim//4)
        self.ln12 = nn.Linear(input_dim//4, input_dim//4)
        self.ln13 = nn.Linear(input_dim//4, input_dim//2)
        self.ln14 = nn.Linear(input_dim//2, input_dim//2)
        self.ln15 = nn.Linear(input_dim//2, output_dim)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(x))
        x = F.relu(self.ln4(x))
        x = F.relu(self.ln5(x))
        x = F.relu(self.ln6(x))
        x = F.relu(self.ln7(x))
        x = F.relu(self.ln8(x))
        x = F.relu(self.ln9(x))
        x = F.relu(self.ln10(x))
        x = F.relu(self.ln11(x))
        x = F.relu(self.ln12(x))
        x = F.relu(self.ln13(x))
        x = F.relu(self.ln14(x))
        x = self.ln15(x)
        return x
