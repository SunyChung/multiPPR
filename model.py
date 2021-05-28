import pickle
import os
import torch
import torch.nn as nn

from layers import ItemLinear


class ContextualizedNN(nn.Module):
    def __init__(self, data_dir, item_cxt_dict, input_dim, hidden_dim, output_dim, top_k):
        super(ContextualizedNN, self).__init__()

        with open(os.path.join(data_dir, 'per_user_item.dict'), 'rb') as f:
            self.per_user_item_dict = pickle.load(f)
        self.item_cxt_dict = item_cxt_dict

        self.top_k = top_k
        self.com_1 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_2 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_3 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_4 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_5 = ItemLinear(input_dim, hidden_dim, output_dim)

        self.interact_linear = nn.Linear(output_dim, 1)

    def forward(self, user_idxs, item_idxs):
        item_rep_list = []
        for i in range(len(item_idxs)):
            item_rep = torch.cat((
                self.com_1(torch.Tensor(self.item_cxt_dict[item_idxs[i]][0]).unsqueeze(1),
                           torch.Tensor(self.item_cxt_dict[item_idxs[i]][1]).unsqueeze(1)),

                self.com_2(torch.Tensor(self.item_cxt_dict[item_idxs[i]][2]).unsqueeze(1),
                           torch.Tensor(self.item_cxt_dict[item_idxs[i]][3]).unsqueeze(1)),

                self.com_3(torch.Tensor(self.item_cxt_dict[item_idxs[i]][4]).unsqueeze(1),
                           torch.Tensor(self.item_cxt_dict[item_idxs[i]][5]).unsqueeze(1)),

                self.com_4(torch.Tensor(self.item_cxt_dict[item_idxs[i]][6]).unsqueeze(1),
                           torch.Tensor(self.item_cxt_dict[item_idxs[i]][7]).unsqueeze(1)),

                self.com_5(torch.Tensor(self.item_cxt_dict[item_idxs[i]][8]).unsqueeze(1),
                           torch.Tensor(self.item_cxt_dict[item_idxs[i]][9]).unsqueeze(1))
                ))
            item_rep_list.append(item_rep)

        batch_user_item_list = [self.per_user_item_dict[i] for i in user_idxs]
        batch_user_item_rep = []
        for i in range(len(batch_user_item_list)):
            per_user_item_rep = []
            for j in range(len(batch_user_item_list[i])):
                item_rep = torch.cat((
                    self.com_1(torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][0]).unsqueeze(1),
                               torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][1]).unsqueeze(1)),

                    self.com_2(torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][2]).unsqueeze(1),
                               torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][3]).unsqueeze(1)),

                    self.com_3(torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][4]).unsqueeze(1),
                               torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][5]).unsqueeze(1)),

                    self.com_4(torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][6]).unsqueeze(1),
                               torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][7]).unsqueeze(1)),

                    self.com_5(torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][8]).unsqueeze(1),
                               torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][9]).unsqueeze(1))
                    ))
                per_user_item_rep.append(item_rep)
            batch_user_item_rep.append(torch.mean(torch.stack(per_user_item_rep), dim=0))

        interaction_list = [batch_user_item_rep[i] * item_rep_list[i] for i in range(len(batch_user_item_rep))]
        # print('interaction_list length : ', len(interaction_list))  #  100
        # 생각해 보니 interaction 은 torch.cat() 안 해도 되겠네 ....
        batch_prediction = [self.interact_linear(interaction_list[i]) for i in range(len(interaction_list))]
        print('batch_prediction length : ', len(batch_prediction))
        result = torch.cat([torch.sigmoid(batch_prediction[i]) for i in range(len(batch_prediction))])
        print('result shape : ', result.shape)
        return result
