import torch

from layers import *
from utils import get_user_sequences


class ContextualizedNN(nn.Module):
    def __init__(self, data_dir, input_dim, hidden_dim, output_dim, top_k):
        super(ContextualizedNN, self).__init__()

        with open(os.path.join(data_dir, 'per_user_item.dict'), 'rb') as f:
            self.per_user_item_dict = pickle.load(f)

        self.top_k = top_k
        self.com_1 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_2 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_3 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_4 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_5 = ItemLinear(input_dim, hidden_dim, output_dim)

        self.interact_linear = nn.Linear(output_dim * 5, 1)
        self.CFs = ContextFeatures(data_dir, top_k)

    def forward(self, user_idxs, item_idxs):
        item_feature = self.CFs.neighbor_embeds(item_idxs)
        item_rep_list = []
        for i in range(len(item_idxs)):
            item_rep = torch.cat((self.com_1(item_feature[0][self.top_k * i:self.top_k * (i+1)],
                                             item_feature[1][self.top_k * i:self.top_k * (i+1)]),
                                  self.com_2(item_feature[2][self.top_k * i:self.top_k * (i+1)],
                                             item_feature[3][self.top_k * i:self.top_k * (i+1)]),
                                  self.com_3(item_feature[4][self.top_k * i:self.top_k * (i+1)],
                                             item_feature[5][self.top_k * i:self.top_k * (i+1)]),
                                  self.com_4(item_feature[6][self.top_k * i:self.top_k * (i+1)],
                                             item_feature[7][self.top_k * i:self.top_k * (i+1)]),
                                  self.com_5(item_feature[8][self.top_k * i:self.top_k * (i+1)],
                                             item_feature[9][self.top_k * i:self.top_k * (i+1)])
                                  ))
            item_rep_list.append(item_rep)
        # 여기서는 user_idx 가 복수이기 때문에, 먼저, user 별 item list 만들고,
        # 각각의 item list 를 위에서 처럼 합쳐서 user representation 만들어야 함
        batch_user_item_list = [self.per_user_item_dict[i] for i in user_idxs]
        batch_user_item_rep = []
        for i in range(len(batch_user_item_list)):
            per_user_item_rep = []
            per_user_item_features = self.CFs.neighbor_embeds(batch_user_item_list[i])
            for j in range(len(batch_user_item_list[i])):
                item_rep =  torch.cat((
                        self.com_1(per_user_item_features[0][self.top_k * j:self.top_k * (j + 1)],
                                   per_user_item_features[1][self.top_k * j:self.top_k * (j + 1)]),
                        self.com_2(per_user_item_features[2][self.top_k * j:self.top_k * (j + 1)],
                                   per_user_item_features[3][self.top_k * j:self.top_k * (j + 1)]),
                        self.com_3(per_user_item_features[4][self.top_k * j:self.top_k * (j + 1)],
                                   per_user_item_features[5][self.top_k * j:self.top_k * (j + 1)]),
                        self.com_4(per_user_item_features[6][self.top_k * j:self.top_k * (j + 1)],
                                   per_user_item_features[7][self.top_k * j:self.top_k * (j + 1)]),
                        self.com_5(per_user_item_features[8][self.top_k * j:self.top_k * (j + 1)],
                                   per_user_item_features[9][self.top_k * j:self.top_k * (j + 1)])
                    ))
                per_user_item_rep.append(item_rep)
            batch_user_item_rep.append(torch.mean(torch.stack(per_user_item_rep), dim=0))
        # 왜 길이랑 type 이 동일한데 interaction type 이 object 인지 ?!
        # 근데, 이렇게 무식하게 list 써도 되는 건가 -_ ;;;
        interaction_list = [batch_user_item_rep[i] * item_rep_list[i] for i in range(len(batch_user_item_rep))]
        batch_prediction = [self.interact_linear(interaction_list[i]) for i in range(len(interaction_list))]
        return torch.cat([torch.sigmoid(batch_prediction[i]) for i in range(len(batch_prediction))])


if __name__ == '__main__':
    data_dir = './data/ml-1m/'
    _, per_user_item_dict = get_user_sequences(data_dir)
    with open(data_dir + 'per_user_item.dict', 'wb') as f:
        pickle.dump(per_user_item_dict, f)
