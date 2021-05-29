import pickle
import os
import torch
import torch.nn as nn

from layers import ItemLinear


class ContextualizedNN(nn.Module):
    def __init__(self, data_dir, item_cxt_dict, input_dim, hidden_dim, output_dim, top_k):
        super(ContextualizedNN, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(os.path.join(data_dir, 'per_user_item.dict'), 'rb') as f:
            self.per_user_item_dict = pickle.load(f)
        self.item_cxt_dict = item_cxt_dict

        self.com_1 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_2 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_3 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_4 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_5 = ItemLinear(input_dim, hidden_dim, output_dim)  # 1 -> 100 -> 10

        self.interact_linear = nn.Linear(output_dim * 5, 1)  # 5 factor output -> final output

    def forward(self, user_idxs, item_idxs):
        batch_item_rep_list = []
        for i in range(len(item_idxs)):
            item_rep = torch.cat((
                self.com_1(torch.Tensor(self.item_cxt_dict[item_idxs[i]][0]).to(self.device),
                           torch.Tensor(self.item_cxt_dict[item_idxs[i]][1]).to(self.device)),

                self.com_2(torch.Tensor(self.item_cxt_dict[item_idxs[i]][2]).to(self.device),
                           torch.Tensor(self.item_cxt_dict[item_idxs[i]][3]).to(self.device)),

                self.com_3(torch.Tensor(self.item_cxt_dict[item_idxs[i]][4]).to(self.device),
                           torch.Tensor(self.item_cxt_dict[item_idxs[i]][5]).to(self.device)),

                self.com_4(torch.Tensor(self.item_cxt_dict[item_idxs[i]][6]).to(self.device),
                           torch.Tensor(self.item_cxt_dict[item_idxs[i]][7]).to(self.device)),

                self.com_5(torch.Tensor(self.item_cxt_dict[item_idxs[i]][8]).to(self.device),
                           torch.Tensor(self.item_cxt_dict[item_idxs[i]][9]).to(self.device))
                ))
            # print('item_rep shape : ', item_rep.shape)  # torch.Size([50])
            batch_item_rep_list.append(item_rep)
        # 각각의 item index 에 대한 dimension 은 10 (=ouput_dim) x 5 (=factor) = 50
        # batch_item_rep_list 는 batch size 만큼의 lenght = 100 를 가지는 게 맞음 !
        # print('batch_item_rep_list length : ', len(batch_item_rep_list))  # 100

        # get item sequence for each user
        batch_user_item_list = [self.per_user_item_dict[i] for i in user_idxs]
        # batch user representation list
        batch_user_item_rep_list = []
        for i in range(len(batch_user_item_list)):
            per_user_item_list = []
            for j in range(len(batch_user_item_list[i])):
                item_rep = torch.cat((
                    # item_cxt_dict 자체가 20 개 값을 반환 : 이 값이 20 개의 idx 값
                    self.com_1(torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][0]).to(self.device),
                               torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][1]).to(self.device)),

                    self.com_2(torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][2]).to(self.device),
                               torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][3]).to(self.device)),

                    self.com_3(torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][4]).to(self.device),
                               torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][5]).to(self.device)),

                    self.com_4(torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][6]).to(self.device),
                               torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][7]).to(self.device)),

                    self.com_5(torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][8]).to(self.device),
                               torch.Tensor(self.item_cxt_dict[batch_user_item_list[i][j]][9]).to(self.device))
                    ))
                # print('item_rep shape : ', item_rep.shape)  # torch.Size([50])
                per_user_item_list.append(item_rep)
            per_user_item_rep = torch.mean(torch.stack(per_user_item_list), dim=0)
            # print('per_user_item_rep shape : ', per_user_item_rep.shape)  # torch.Size([50])
            batch_user_item_rep_list.append(per_user_item_rep)
        # 정작 필요한 건 user, item representation 인데,
        # 이게 batch 로 모여 있어서, 중간에 저장할 방법 찾아야 ...

        interaction_list = [batch_user_item_rep_list[i] * batch_item_rep_list[i]
                            for i in range(len(batch_user_item_rep_list))]
        # print('interaction_list length : ', len(interaction_list))  # 100
        # print('interaction_list[0] shape : ', interaction_list[0].shape)  # torch.Size([50])
        batch_prediction = [torch.sigmoid(self.interact_linear(interaction_list[i]))
                            for i in range(len(interaction_list))]
        # print('batch_prediction[0] shape : ', batch_prediction[0].shape)  # torch.Size([1])
        result = torch.stack(batch_prediction).squeeze(1)
        # print('result shape : ', result.shape)  # torch.Size([100])
        # 결과적으로는 batch size 만큼의 (=100) prediction 값을 반환해야 함
        return result
