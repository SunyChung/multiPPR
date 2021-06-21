import os
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from multi_model import ContextualizedNN
from features import PPRfeatures
from utils import load_all, get_sparse_coord_value_shape


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/ml-1m/')
# parser.add_argument('--data_dir', type=str, default='./data/ml-20m/')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
# 5e-4, 1e-4, 1e-3
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--multi_factor', type=int, default=5)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--emb_dim', type=int, default=10)

args = parser.parse_args()
data_dir = args.data_dir
learning_rate = args.learning_rate
batch_size = args.batch_size
# epochs = args.epochs
# epochs = 10
epochs = 1
print('epochs : ', epochs)
multi_factor = args.multi_factor
# top_k = args.top_k # top_k = 15  # top_k = 5
# top_k = 100  # RuntimeError: CUDA out of memory.
top_k = 50
print('extracting top-k : ', top_k)
# emb_dim = args.emb_dim
emb_dim = 100
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
print('n_items : ', n_items)
item_embedding = nn.Embedding(n_items, emb_dim)
user_embedding = nn.Embedding(len(unique_uidx), emb_dim)

model = ContextualizedNN(item_idx_tensor, item_scr_tensor, user_idx_tensor, user_scr_tensor,
                         item_embedding, user_embedding, multi_factor, top_k).to(device)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable parameters : ', pytorch_total_params)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print('learning rate : ', learning_rate)


def NDCG(pred_rel, item_idxs, k):
    topk_idx = np.argsort(pred_rel)[::-1][:k]
    topk_pred = pred_rel[topk_idx]
    # print('topk_pred : ', topk_pred)  # 전부 1 만 ....;
    # 그래서 dcg 값이 idcg 값보다 커지고 있음 -_
    discount = 1. / np.log2(np.arange(2, k + 2))
    dcg = np.array(topk_pred * discount).sum()
    print('dcg : ', dcg)
    # max() 로 하니 item_idx 길이랑 k 가 같아지면서
    # dcg / idcg = 1 값이 나옴
    # 다시 한 번, prediction 값이 모두 1 이라서 생기는 결과임을 확인 !
    # 결론은 dataset 을 0 과 1 이 모두 포함되게 바꾸거나
    # positive-negative pair 로 훈련시켜야 함
    idcg = np.array(discount[:max(len(item_idxs), k)].sum())
    print('idcg : ', idcg)
    print('dcg / idcg : ', dcg / idcg)
    return dcg / idcg


def RECALL(pred_rel, item_idx, k):
    # print('true item length : ', len(item_idx))
    topk_idx = np.argsort(pred_rel)[::-1][:k]
    pred_binary = np.zeros_like(pred_rel, dtype=bool)
    pred_binary[topk_idx] = True
    true_binary = np.zeros_like(pred_rel, dtype=bool)
    true_binary[item_idx] = True
    tmp = (np.logical_and(pred_binary, true_binary).sum()).astype(np.float32)
    recall = tmp / np.minimum(k, true_binary.sum())
    return recall


def evaluate(test_input, n_items):
    model.eval()
    row, col = test_input.nonzero()
    uniq_users = np.unique(row)
    uniq_items = np.unique(col)
    input_array = test_input.toarray()
    recall_list = []
    ndcg_list = []
    for i in range(len(uniq_users)):
        user_idxs = np.repeat(uniq_users[i], n_items)
        item_idxs = range(n_items)
        predictions = model(user_idxs, item_idxs).detach().cpu().numpy()
        # print('predictions : ', predictions)
        target_user_items = np.where(input_array[uniq_users[i], :] == 1)[0]

        ndcg_score = NDCG(predictions, target_user_items, k=100)
        # print('ndcg_score ', ndcg_score)
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
        # 근데, 여기서 문제가 target 이 다 1 이라는 거 -_ 이렇게 훈련하면 당연히 1 로만 수렴하게 됨 !
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
