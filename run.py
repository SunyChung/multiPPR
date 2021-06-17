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


def evaluate(test_input, n_items):
    model.eval()
    row, col = test_input.nonzero()
    uniq_users = np.unique(row)
    uniq_items = np.unique(col)
    input_array = test_input.toarray()
    recall_list = []
    ndcg_list = []
    for i in range(len(uniq_users)):
        # 여기서 자꾸 문제가 뭐냐면, 사실 evluation 을 위해 맞춰야 하는 정답은
        # target_user_items 인데,
        # model 은 전체 item 인 n_items 에 대해서 prediction 을 하고 있음
        # uniq_items 로만 prediction 하면, 좀 달라지나 ?
        # 그리고 왜 prediction 은 전부 동일한 숫자만 찍는 건지 ?! -_
        # pred_rel :  [0.41123354 0.41123354 0.41123354 ... 0.41123354 0.41123354 0.41123354]
        predictions = model(np.repeat(uniq_users[i], len(uniq_items)), uniq_items).detach().cpu().numpy()
        target_user_items = np.where(input_array[uniq_users[i], :] == 1)[0]

        ndcg_score = NDCG(predictions, target_user_items, k=100)
        ndcg_list.append(ndcg_score)
        recall_score = RECALL(predictions, target_user_items, k=100)
        recall_list.append(recall_score)
    return ndcg_list, recall_list


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
    ndcg_list, recall_list = evaluate(valid_input, n_items)
    return ndcg_list, recall_list


for epoch in range(epochs):
    ndcg_list, recall_list = train(epoch, train_data, vad_data)
    print('returned NDCG : ', np.mean(ndcg_list))
    print('returned recall :', np.mean(recall_list))

print('test started !')
ndcg_list, recall_list = evaluate(test_data, n_items)
print('returned NDCG : ', np.mean(ndcg_list))
print('returned recall :', np.mean(recall_list))
