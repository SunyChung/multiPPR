import os
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from multi_model import ContextualizedNN
# from model import ContextualizedNN
from features import PPRfeatures
from utils import load_all

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/ml-1m/')
# parser.add_argument('--data_dir', type=str, default='./data/ml-20m/')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')  # 5e-4, 1e-4, 1e-3
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--multi_factor', type=int, default=5)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--emb_dim', type=int, default=10)
parser.add_argument('--damping', type=float, default=0.85)

args = parser.parse_args()
data_dir = args.data_dir
learning_rate = args.learning_rate
# batch_size = args.batch_size
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
damping = args.damping

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# item context tensor
# multi-factor features
with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
    item_idx_dict = pickle.load(f)
with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
    item_ppr_dict = pickle.load(f)
pf = PPRfeatures(data_dir, top_k, item_idx_dict, item_ppr_dict)
# one-factor features
# with open(os.path.join(data_dir, 'per_item_idx'+ str(damping) + '.dict'), 'rb') as f:
#     item_idx_dict = pickle.load(f)
# with open(os.path.join(data_dir, 'per_item_ppr'+ str(damping) + '.dict'), 'rb') as f:
#     item_ppr_dict = pickle.load(f)
# pf = OneFactorFeature(data_dir, top_k, item_idx_dict, item_ppr_dict)
item_idx_tensor = pf.idx_tensor()
item_scr_tensor = pf.score_tensor()
del item_idx_dict
del item_ppr_dict
# user context tensor
# multi-factor features
with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
    user_idx_dict = pickle.load(f)
with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
    user_ppr_dict = pickle.load(f)
pf = PPRfeatures(data_dir, top_k, user_idx_dict, user_ppr_dict)
# one-factor features
# with open(os.path.join(data_dir, 'per_user_idx'+ str(damping) + '.dict'), 'rb') as f:
#    user_idx_dict = pickle.load(f)
# with open(os.path.join(data_dir, 'per_user_ppr'+ str(damping) + '.dict'), 'rb') as f:
#    user_ppr_dict = pickle.load(f)
# pf = OneFactorFeature(data_dir, top_k, user_idx_dict, user_ppr_dict)
user_idx_tensor = pf.idx_tensor()
user_scr_tensor = pf.score_tensor()
del user_idx_dict
del user_ppr_dict

# for user embedding size
with open(os.path.join(data_dir, 'unique_uidx'), 'rb') as f:
    unique_uidx = pickle.load(f)
n_items, train_coords, train_values, vad_coords, vad_values, test_coords, test_values = load_all(data_dir)
batch_size = n_items
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


def NDCG(predictions, targets, k):
    topk_idx = np.argsort(predictions)[::-1][:min(len(predictions), k)]
    topk_pred = predictions[topk_idx]
    discount = 1. / np.log2(np.arange(2, min(len(predictions), k) + 2))
    dcg = np.array(topk_pred * discount).sum()
    # print('dcg : ', dcg)
    idcg = np.array(discount[:min(len(targets), k)].sum())
    # print('idcg : ', idcg)
    # print('dcg / idcg : ', dcg / idcg)
    return dcg / idcg


def RECALL(predictions, targets, k):
    # print('true item length : ', len(item_idx))
    topk_idx = np.argsort(predictions)[::-1][:min(len(predictions), k)]
    pred_binary = np.zeros_like(predictions, dtype=bool)
    pred_binary[topk_idx] = True
    tmp = (np.logical_and(pred_binary, targets).sum()).astype(np.float32)
    # print('targets : ', targets)
    print('target sum : ', targets.sum())  # 왜 다 0 이지 ?!
    # 이것 때문에라도 data shape 바꿔야 ...
    # 근데, 이 정도로 sparse 한가 ?!
    dinorm = min(k, targets.sum())
    recall = tmp / dinorm
    return recall


def evaluate(test_coords, test_values):
    model.eval()
    print(len(test_coords))  #
    recall_list = []
    ndcg_list = []
    test_n = len(test_coords)
    test_idxlist = list(range(test_n))
    for batch_num, st_idx in enumerate(range(0, test_n, batch_size)):
        end_idx = min(st_idx + batch_size, test_n)
        user_idxs = test_coords[test_idxlist[st_idx:end_idx]][:, 0]
        item_idxs = test_coords[test_idxlist[st_idx:end_idx]][:, 1]
        predictions = model(user_idxs, item_idxs).detach().cpu().numpy()
        # print('predictions : ', predictions)
        targets = test_values[test_idxlist[st_idx:end_idx]]
        # print('targets : ', targets)
        # print('target shape : ', targets.shape)
        ndcg_score = NDCG(predictions, targets, k=100)
        ndcg_list.append(ndcg_score)
        recall_score = RECALL(predictions, targets, k=100)
        recall_list.append(recall_score)
    return ndcg_list, recall_list


def train(epoch, train_coords, train_values):
    print('epoch : ', epoch)
    start = time.time()
    print(len(train_coords))  # 21102072
    train_n = len(train_coords)
    train_idxlist = list(range(train_n))
    model.train()
    optimizer.zero_grad()
    loss = nn.BCELoss()
    # loss = BPRLoss()
    for batch_num, st_idx in enumerate(range(0, train_n, batch_size)):
        end_idx = min(st_idx + batch_size, train_n)
        user_idxs = train_coords[train_idxlist[st_idx:end_idx]][:, 0]
        item_idxs = train_coords[train_idxlist[st_idx:end_idx]][:, 1]
        predictions = model(user_idxs, item_idxs)
        targets = torch.Tensor(train_values[train_idxlist[st_idx:end_idx]])
        train_loss = loss(predictions.to('cpu'), targets)
        train_loss.backward()
        optimizer.step()
    print('one epoch training takes : ', time.time() - start)
    ndcg_list, recall_list = evaluate(vad_coords, vad_values)
    return ndcg_list, recall_list


for epoch in range(epochs):
    ndcg_list, recall_list = train(epoch, train_coords, train_values)
    print('returned NDCG : ', np.mean(ndcg_list))
    print('returned recall :', np.mean(recall_list))

print('test started !')
ndcg_list, recall_list = evaluate(test_coords, test_values)
print('returned NDCG : ', np.mean(ndcg_list))
print('returned recall :', np.mean(recall_list))
