import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import ContextualizedNN
from layers import PPRfeatures
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/ml-1m/')
# parser.add_argument('--data_dir', type=str, default='./data/ml-20m/')
parser.add_argument('--damping_factors', type=list, default=[0.30, 0.50, 0.70, 0.85, 0.95])
parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
# 5e-4, 1e-4, 1e-3
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--multi_factor', type=int, default=5)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--emb_dim', type=int, default=10)

args = parser.parse_args()
data_dir = args.data_dir
damping_factors = args.damping_factors
learning_rate = args.learning_rate
batch_size = args.batch_size
# epochs = args.epochs
epochs = 10
print('epochs : ', epochs)
multi_factor = args.multi_factor
# top_k = args.top_k # top_k = 15  # top_k = 5
# top_k = 100  # RuntimeError: CUDA out of memory.
top_k =  50
print('extracting top-k features : ', top_k)
# emb_dim = args.emb_dim
emb_dim = 150
print('embedding dimension : ', emb_dim)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# item context tensor
with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
    item_idx_dict = pickle.load(f)
with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
    item_ppr_dict = pickle.load(f)

pf = PPRfeatures(data_dir, top_k, item_idx_dict, item_ppr_dict)
item_idx_tensor = pf.idx_tensor()
item_scr_tensor = pf.score_tensor()
del item_idx_dict
del item_ppr_dict

# user context tensor
with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
    user_idx_dict = pickle.load(f)
with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
    user_ppr_dict = pickle.load(f)

pf = PPRfeatures(data_dir, top_k, user_idx_dict, user_ppr_dict)
user_idx_tensor = pf.idx_tensor()
user_scr_tensor = pf.score_tensor()
del user_idx_dict
del user_ppr_dict

# for user embedding size
with open(os.path.join(data_dir, 'unique_uidx'), 'rb') as f:
    unique_uidx = pickle.load(f)

n_items, train_data, vad_data, test_data = load_all(data_dir)
item_embedding = nn.Embedding(n_items, emb_dim)
user_embedding = nn.Embedding(len(unique_uidx), emb_dim)

model = ContextualizedNN(item_idx_tensor, item_scr_tensor, user_idx_tensor, user_scr_tensor,
                         item_embedding, user_embedding).to(device)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable parameters : ', pytorch_total_params)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print('learning rate : ', learning_rate)


def evaluate(test_input):
    model.eval()
    row, col = test_input.nonzero()
    uniq_users = np.unique(row)
    user_idx = np.where(uniq_users[:-1] != uniq_users[1:])[0] + 1
    user_idx = np.insert(user_idx, 0, 0)
    uniq_items = np.unique(col)
    input_array = test_input.toarray()
    recall_list = []
    ndcg_list = []
    for i in range(len(user_idx)):
        st_idx = user_idx[i]
        ed_idx = user_idx[i+1]
        predictions = model(row[st_idx:ed_idx], col[st_idx:ed_idx]).detach().cpu().numpy()
        # target_user_items = np.where(input_array[uniq_users[i], :] == 1)[0]
        target_user_items = col[st_idx:ed_idx]
        ndcg_score = NDCG(predictions, target_user_items, k=100)
        ndcg_list.append(ndcg_score)
        recall_score = RECALL(predictions, target_user_items, k=100)
        recall_list.append(recall_score)
    return ndcg_list, recall_list


'''
def evaluate(data_tr, data_te, n_items):
    model.eval()
    tr_row, tr_col = data_tr.nonzero()
    te_row, te_col = data_te.nonzero()
    tr_users = np.unique(tr_row)
    tr_items = np.unique(tr_col)
    te_items = np.unique(te_col)
    
    
    batch_uniq_users = np.repeat(uniq_users, n_items)
    # np.repeat() repeats the elements of an array
    # 499 x 3503 = (1747997,)
    batch_n_items = np.tile(np.array(range(n_items)), len(uniq_users))
    # whereas np.tile() repeat the whole array n-times
    # 3503 x 499 = (1747997,)
    # 이 경우에는 몇 개씩 데이터를 넣고, 예측값은 몇 개씩 나눠야 하나 ?!
    # 3503 개를 하나의 묶음으로 봐야 하긴 함
    # 근데, n_items 는 user 갯수만큼 반복해야지 dimension 이 맞을 듯 !
    predictions = model(batch_uniq_users, batch_n_items)
    # 여기서 정작 중요한 건 batch prediction 이 아닌데 ?!
    # 다시 보니 아까 넣은 target item index 값도 잘 넣었는데 ...
    test_array = test_data.toarray()
'''


def train(epoch, train_input, valid_input):
    print('epoch : ', epoch)
    start = time.time()

    train_coords, train_values, _ = get_sparse_coord_value_shape(train_input)
    # print(train_coords.shape)  # (477497, 2)
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
    print('one epoch training takes : ', time.time() - start)
    # evaluation with train data set
    ndcg_list, recall_list = evaluate(valid_input)
    return ndcg_list, recall_list


for epoch in range(epochs):
    ndcg_list, recall_list = train(epoch, train_data, vad_data)
    print('returned NDCG : ', np.mean(ndcg_list))
    print('returned recall :', np.mean(recall_list))

print('test started !')
ndcg_list, recall_list = evaluate(test_data)
print('returned NDCG : ', np.mean(ndcg_list))
print('returned recall :', np.mean(recall_list))
