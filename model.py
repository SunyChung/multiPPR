from layers import *
from utils import get_user_sequences


class ContextualizedNN(nn.Module):
    def __init__(self, data_dir, input_dim, hidden_dim, output_dim, top_k):
        super(ContextualizedNN, self).__init__()

        with open(os.path.join(data_dir, 'per_user_item.dict'), 'rb') as f:
            self.per_user_item_dict = pickle.load(f)

        self.com_1 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_2 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_3 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_4 = ItemLinear(input_dim, hidden_dim, output_dim)
        self.com_5 = ItemLinear(input_dim, hidden_dim, output_dim)

        self.interact_linear = nn.Linear(output_dim * 5, 1)
        self.CFs = ContextFeatures(data_dir, top_k)

    def forward(self, user_idxs, item_idxs):
        feature_list = self.CFs.neighbor_embeds(target_idx=item_idxs)
        # ('feature_list len : ', len(feature_list))
        # print('feature_list[0] : ', feature_list[0])
        # tensor([1195.,  259., 2857., 1197., 2570.,  592., 2027.,  317., 1269., 2761.,
        # 1196., 1209., 1264.,  526.,  588.,  607., 2395., 3113.,  295.,  109.])
        # print('feature_list[0] type : ', type(feature_list))  #  <class 'list'>
        # print('feature_list[0][0] : ', feature_list[0][0])  # tensor(1195.)

        item_rep = torch.cat((self.com_1(feature_list[0], feature_list[1]),
                              self.com_2(feature_list[2], feature_list[3]),
                              self.com_3(feature_list[4], feature_list[5]),
                              self.com_4(feature_list[6], feature_list[7]),
                              self.com_5(feature_list[8], feature_list[9])
        ))

        per_user_items = self.per_user_item_dict[user_idxs]
        item_list = []
        for i in per_user_items:
            feat = self.CFs.neighbor_embeds(target_idx=i)
            item = torch.cat((self.com_1(feat[0], feat[1]),
                              self.com_2(feat[2], feat[3]),
                              self.com_3(feat[4], feat[5]),
                              self.com_4(feat[6], feat[7]),
                              self.com_5(feat[8], feat[9])
            ))
            item_list.append(item)
        user_rep = torch.mean(torch.stack(item_list), dim=0)
        # print('user_rep size : ', user_rep.shape)  # user_rep size :  torch.Size([50])
        # print('item_rep size : ', item_rep.shape) # item_rep size :  torch.Size([50]) = 10 x 5
        interaction = user_rep * item_rep
        # print('interaction size : ', interaction.shape)  # interaction size :  torch.Size([50])
        prediction = self.interact_linear(interaction)
        return torch.sigmoid(prediction)


if __name__ == '__main__':
    data_dir = './data/ml-1m/'
    _, per_user_item_dict = get_user_sequences(data_dir)
    with open(data_dir + 'per_user_item.dict', 'wb') as f:
        pickle.dump(per_user_item_dict, f)

    top_k = 20
    input_dim = top_k
    hidden_dim = 40
    output_dim = 10

