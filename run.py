import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model import ContextualizedNN
from layers import PPRfeatures
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/ml-1m/')
parser.add_argument('--damping_factors', type=list, default=[0.30, 0.50, 0.70, 0.85, 0.95])
parser.add_argument('--learning_rate', type=float, default=5e-4, help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--multi_factor', type=int, default=5)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--input_dim', type=int, default=20, help='top_k x multi_factor')
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--output_dim', type=int, default=10)
parser.add_argument('--final_dim', type=int, default=1)

args = parser.parse_args()
data_dir = args.data_dir
damping_factors = args.damping_factors
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
multi_factor = args.multi_factor
top_k = args.top_k
input_dim = top_k * multi_factor
hidden_dim = input_dim * 5
output_dim = multi_factor * 10
final_dim = args.final_dim

# item context dictionary making
with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
    item_idx_dict = pickle.load(f)
with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
    item_ppr_dict = pickle.load(f)
pf = PPRfeatures(data_dir, top_k, item_idx_dict, item_ppr_dict)
item_idx_emb = pf.idx_embeds()
item_scr_emb = pf.score_embeds()
del item_idx_dict
del item_ppr_dict

# user context dictionary making
with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
    user_idx_dict = pickle.load(f)
with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
    user_ppr_dict = pickle.load(f)
pf = PPRfeatures(data_dir, top_k, user_idx_dict, user_ppr_dict)
user_idx_emb = pf.idx_embeds()
user_scr_emb = pf.score_embeds()
del user_idx_dict
del user_ppr_dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te = load_data(data_dir)
model = ContextualizedNN(item_idx_emb, item_scr_emb, user_idx_emb, user_scr_emb,
                         multi_factor, input_dim, hidden_dim, output_dim, final_dim).to(device)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable parameters : ', pytorch_total_params)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_coords, train_values, _ = get_sparse_coord_value_shape(train_data)
vad_tr_coords, vad_trv_values, _ = get_sparse_coord_value_shape(vad_data_tr)
vad_te_coords, vad_te_values, _ = get_sparse_coord_value_shape(vad_data_te)
# print(train_coords)  # user, item index lists
# print(train_coords.shape)  # (480722, 2)
# print(train_values)


# RecVAE, VAE_cf 랑 학습 방식에 차이가 있음 !
# VAE-based 방식은 user 별로 전체 item 에 대한 interaction probability 를 학습하는 구조고,
# 내 모델은 주어진 user-item pair index 를 사용해, 각각의 representaiton 을 학습함
# 일단, train_data 에 포함된 item 에 대해서만, 학습이 이루어지기 때문에,
# train_data 로 학습한 item representation 은 validation, test 에서도 사용 가능
# 대신, user representation 은 PPR 을 사용해, 유추하는 방식으로, (이것도 될 것 같긴 함...)
# 근데, 이 방식에서는 validation 에서 validation train 을 validation test 에 적용 가능한가?
# train, test index 부터 다른 거 같은데 -_
# 그냥 validation test 로만 성능 평가해도 될 듯 ....
def train(epoch, train_coords, train_values, vad_te_coords, vad_te_values):
    print('epoch : ', epoch)
    start = time.time()

    train_n = len(train_coords)
    train_idxlist = list(range(train_n))
    model.train()
    optimizer.zero_grad()
    loss = nn.BCELoss()
    for batch_num, st_idx in enumerate(range(0, train_n, batch_size)):
        # print('batch num : ', batch_num)
        end_idx = min(st_idx + batch_size, train_n)
        user_idxs = train_coords[train_idxlist[st_idx:end_idx]][:, 0]
        item_idxs = train_coords[train_idxlist[st_idx:end_idx]][:, 1]

        predictions = model(user_idxs, item_idxs)
        targets = torch.Tensor(train_values[train_idxlist[st_idx:end_idx]])
        train_loss = loss(predictions.to('cpu'), targets)
        train_loss.backward()
        optimizer.step()

    vad_n = len(vad_te_coords)
    vad_idxlist = list(range(vad_n))
    ndcg_list = []
    for batch_num, st_idx in enumerate(range(0, vad_n, batch_size)):
        end_idx = min(st_idx + batch_size, vad_n)
        user_idxs = vad_te_coords[vad_idxlist[st_idx:end_idx]][:, 0]
        item_idxs = vad_te_coords[vad_idxlist[st_idx:end_idx]][:, 1]

        predictions = model(user_idxs, item_idxs)
        ndcg_list.append(NDCG(predictions, vad_te_values[vad_idxlist[st_idx:end_idx]]))
    ndcg_list = np.concatenate(ndcg_list)
    ndcg = ndcg_list.mean()
    print('epoch : {:04d}'.format(epoch),
          '\ntime : {:.4f}s'.format(time.time() - start))
    return ndcg


for epoch in range(epochs):
    ndcg = train(epoch, train_coords, train_values, vad_te_coords, vad_te_values)
    print('NDCG : ', ndcg)


# 근데, 이렇게 되면, 예측이 너무 뻔해지는데;;
# 그냥 1 로 다 찍으면 끝나는 거 아닌가 ? ......
def evaluate(test_te_coords, test_te_values):
    test_n = len(test_te_coords)
    idxlist = list(range(test_n))
    model.eval()
    loss = nn.BCELoss()
    loss_list = []
    recall_list = []
    ndcg_list = []
    for batch_num, st_idx in enumerate(range(0, test_n, batch_size)):
        print('batch_num : ', batch_num)
        end_idx = min(st_idx + batch_size, test_n)
        user_idx = test_te_coords[idxlist[st_idx:end_idx]][:, 0]
        item_idx = test_te_coords[idxlist[st_idx:end_idx]][:, 1]

        predictions = model(user_idx, item_idx)
        targets = torch.Tensor(test_te_values[idxlist[st_idx:end_idx]])
        recall_score = RECALL(predictions, targets)
        recall_list.append(recall_score)
        ndcg_score = NDCG(predictions, targets)
        ndcg_list.append(ndcg_score)
        test_loss = loss(predictions.to('cpu'), targets)
        loss_list.append(test_loss.detach())
    return recall_list, ndcg_list, loss_list


print('test started !')
test_tr_coords, test_tr_values, _ = get_sparse_coord_value_shape(test_data_tr)
test_te_coords, test_te_values, _ = get_sparse_coord_value_shape(test_data_te)
returned_loss = evaluate(test_te_coords, test_te_values)
print('returned mean loss : ', np.mean(returned_loss))
print('returned recall :')
