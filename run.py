import os
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import ContextualizedNN
from features import PPRfeatures
from utils import load_all

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/ml-1m/')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--multi_factor', type=int, default=5)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--emb_dim', type=int, default=100)

args = parser.parse_args()
data_dir = args.data_dir
lr = args.learning_rate
# batch_size = args.batch_size
# epochs = args.epochs  #
epochs = 10
multi_factor = args.multi_factor
# top_k = args.top_k
top_k = 10
# emb_dim = args.emb_dim
emb_dim = 64
print('learning rate : ', lr)
print('multi factors : ', multi_factor)
print('epochs : ', epochs)
print('top_k : ', top_k)
print('embedding dimension : ', emb_dim)

n_items, train_data, vad_data, test_data = load_all(data_dir)
print('n_items : ', n_items)

# batch size 가 n_items 가 아니면, NDCG 및 RECALL 이 inf, nan 으로 반환됨 -_
batch_size = n_items
# batch_size = 500
print('batch size : ', batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using {} device'.format(device))
# item feature tensor
with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
    item_idx_dict = pickle.load(f)
with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
    item_ppr_dict = pickle.load(f)
pf = PPRfeatures(top_k, item_idx_dict, item_ppr_dict)
item_idx_tensor = pf.idx_tensor().to(device)
item_scr_tensor = pf.scr_tensor().to(device)
del item_idx_dict
del item_ppr_dict
# user feature tensor
with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
    user_idx_dict = pickle.load(f)
with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
    user_ppr_dict = pickle.load(f)
pf = PPRfeatures(top_k, user_idx_dict, user_ppr_dict)
user_idx_tensor = pf.idx_tensor().to(device)
user_scr_tensor = pf.scr_tensor().to(device)
del user_idx_dict
del user_ppr_dict

with open(os.path.join(data_dir, 'unique_uidx'), 'rb') as f:
    unique_uidx = pickle.load(f)
item_embedding = nn.Embedding(n_items, emb_dim).to(device)
user_embedding = nn.Embedding(len(unique_uidx), emb_dim).to(device)

model = ContextualizedNN(item_idx_tensor, item_scr_tensor,
                         user_idx_tensor, user_scr_tensor,
                         item_embedding, user_embedding).to(device)
print('\n')
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable parameters : ', pytorch_total_params)
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.RMSprop(model.parameters(), lr=lr)
loss = nn.BCELoss()
# loss = nn.BCEWithLogitsLoss()
print('optimizer : ', optimizer)
print('loss function : ', loss)


def train(epoch, train_data, vad_data):
    print('\nepoch : ', epoch)
    start = time.time()
    # print('train batch length : ', len(train_data))
    train_n = len(train_data)
    train_idxlist = list(range(train_n))
    model.train()
    optimizer.zero_grad()

    loss_list = []
    for batch_num, st_idx in enumerate(range(0, train_n, batch_size)):
        # print('batch_num : ', batch_num)
        end_idx = min(st_idx + batch_size, train_n)
        user_idxs = train_data[train_idxlist[st_idx:end_idx]][:, 0]
        item_idxs = train_data[train_idxlist[st_idx:end_idx]][:, 1]
        # print('user_idxs shape : ', user_idxs.shape)  # (3516,)
        # print('item_idxs shape : ', item_idxs.shape)  # (3516,)
        targets = torch.Tensor(train_data[train_idxlist[st_idx:end_idx]][:, 2])
        # print('targets : ', targets)
        # print('target shape : ', targets.shape)
        predictions = model(user_idxs, item_idxs)
        # print('predictions : ', predictions.to('cpu'))
        train_loss = loss(predictions.to('cpu'), targets)
        loss_list.append(train_loss.detach().to('cpu').numpy())
        # print('train_loss : ', train_loss)
        train_loss.backward()
        optimizer.step()
    print('one epoch takes : ', time.time() - start)
    ndcg_100_list, recall_50_list, recall_20_list = evaluate(vad_data)
    return ndcg_100_list, recall_50_list, recall_20_list, loss_list


def NDCG(predictions, targets, k):
    topk_idx = np.argsort(predictions)[::-1][:k]
    # topk_pred = predictions[topk_idx]
    # discount 는 숫자 값 하나가 아니고, (2 ~ k+2) range 의 np.array
    discount = 1. / np.log2(np.arange(2, k+2))
    # DCG = np.array(targets[topk_idx] * discount).sum()
    DCG = np.array(predictions[topk_idx] * discount).sum()
    # print('DCG : ', DCG)
    IDCG = discount[:min(targets.sum(), k)].sum()
    # print('IDCG : ', IDCG)
    # print('DCG / IDCG : ', DCG / IDCG)
    return DCG / IDCG

# 이 정의가 맞는건지?
def RECALL(predictions, targets, k):
    topk_idx = np.argsort(predictions)[::-1][:k]
    pred_binary = np.zeros_like(predictions, dtype=bool)
    pred_binary[topk_idx] = True
    # print('pred_binary : ', pred_binary)
    true_binary = targets > 0
    # print('true_binary : ', true_binary)
    tmp = (np.logical_and(pred_binary, true_binary).sum()).astype(np.float32)
    # print('tmp : ', tmp)
    # print('target sum : ', targets.sum())
    dinorm = min(k, targets.sum())
    # print('dinorm : ', dinorm)
    return tmp / dinorm


def evaluate(test_data):
    model.eval()
    # print('evaluation batch length : ', len(test_data))
    test_n = len(test_data)
    test_idxlist = list(range(test_n))
    ndcg_100_list, recall_50_list, recall_20_list = [], [], []

    for batch_num, st_idx in enumerate(range(0, test_n, batch_size)):
        end_idx = min(st_idx + batch_size, test_n)
        user_idxs = test_data[test_idxlist[st_idx:end_idx]][:, 0]
        item_idxs = test_data[test_idxlist[st_idx:end_idx]][:, 1]
        predictions = model(user_idxs, item_idxs)
        targets = test_data[test_idxlist[st_idx:end_idx]][:, 2]

        ndcg_100_score = NDCG(predictions.detach().to('cpu').numpy(), targets, k=100)
        ndcg_100_list.append(ndcg_100_score)
        recall_50_score = RECALL(predictions.detach().to('cpu').numpy(), targets, k=50)
        recall_20_score = RECALL(predictions.detach().to('cpu').numpy(), targets, k=20)
        recall_50_list.append(recall_50_score)
        recall_20_list.append(recall_20_score)
    return ndcg_100_list, recall_50_list, recall_20_list

# all_epoch_max_100_ndcg, all_epoch_max_50_recall, all_epoch_max_20_recall = [], [], []
for epoch in range(epochs):
    ndcg_100_list, recall_50_list, recall_20_list, loss_list = train(epoch, train_data, vad_data)
    print('\ntraining evaluation ... ')
    print('epoch loss : ', np.mean(loss_list))
    print('mean NDCG@100 : ', np.mean(ndcg_100_list))
    print('mean RECALL@50 : ', np.mean(recall_50_list))
    print('mean RECALL@20 : ', np.mean(recall_20_list))

    # print('max NDCG@100 : ', max(ndcg_100_list))
    # print('max RECALL@50 : ', max(recall_50_list))
    # print('max RECALL@20 : ', max(recall_20_list))

#     all_epoch_max_100_ndcg.append(max(ndcg_100_list))
#     all_epoch_max_50_recall.append(max(recall_50_list))
#     all_epoch_max_20_recall.append(max(recall_20_list))
# print('\nepoch max NDCG@100 : ', max(all_epoch_max_100_ndcg))
# print('epoch max RECALL@50 : ', max(all_epoch_max_50_recall))
# print('epoch max RECALL@20 : ', max(all_epoch_max_20_recall))

#     if epoch == (epochs - 1):
#         with open(os.path.join(data_dir, 'train_loss_with_epoch_' + str(epochs)), 'wb') as f:
#             pickle.dump(loss_list, f)
# with open(os.path.join(data_dir,
#                        'mean_NDCG_with_top_' + str(top_k) + '_epochs_' + str(epochs)), 'wb') as f:
#     pickle.dump(mean_ndcg, f)
# with open(os.path.join(data_dir,
#                        'mean_RECALL_with_top_' + str(top_k) + '_epochs_' + str(epochs)), 'wb') as f:
#     pickle.dump(mean_recall, f)


print('\ntest started !')
ndcg_100_list, recall_50_list, recall_20_list = evaluate(test_data)
print('mean NDCG@100 : ', np.mean(ndcg_100_list))
print('mean RECALL@50 : ', np.mean(recall_50_list))
print('mean RECALL@20 : ', np.mean(recall_20_list))
print('\nmax NDCG@100 : ', max(ndcg_100_list))
print('max RECALL@50 : ', max(recall_50_list))
print('max RECALL@20 : ', max(recall_20_list))

# with open(os.path.join(data_dir,
#                        'test_NDCG_with_top_' + str(top_k) + '_epochs_' + str(epochs)), 'wb') as f:
#     pickle.dump(ndcg_list, f)
# with open(os.path.join(data_dir,
#                        'test_RECALL_with_top_' + str(top_k) + '_epochs_' + str(epochs)), 'wb') as f:
#     pickle.dump(recall_list, f)

print('final item_emb min : ', torch.min(item_embedding.weight))
print('final item_emb mean : ', torch.mean(item_embedding.weight))
print('final item_emb max : ', torch.max(item_embedding.weight))
print('final user_emb min : ', torch.min(user_embedding.weight))
print('final user_emb mean : ', torch.mean(user_embedding.weight))
print('final user_emb max : ', torch.max(user_embedding.weight))
