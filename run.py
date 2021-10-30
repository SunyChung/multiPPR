import os
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from features import get_ppr
from load_data import Data
from utils import *

from oneway_concat_model import oneway_concat

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/gowalla')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--multi_factor', type=int, default=5)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--emb_dim', type=int, default=100)

args = parser.parse_args()
# data_dir = args.data_dir
data_dir = './data/yelp2018'
lr = args.learning_rate
# lr = 1e-3
multi_factor = args.multi_factor
# epochs = 10
epochs = 50
# epochs = 100
# top_k = 50
# top_k = 100
top_k = 20
emb_dim = 64
num_sample = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using {} device'.format(device))
print('learning rate : ', lr)
print('multi factors : ', multi_factor)
print('epochs : ', epochs)
print('top_k : ', top_k)
print('embedding dimension : ', emb_dim)
print('num of train samples : ', num_sample)

dataset = Data(data_dir)
train_pos, train_neg = dataset.make_train_sample_mat(num_sample)
# train_all_mat = dataset.make_train_all_mat()
test_pos, test_neg = dataset.make_test_sample_mat(num_sample)
# test_mat = dataset.make_test_all_mat()
n_users, n_items = dataset.get_num_users_items()

# item feature tensor
with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
    item_idx_dict = pickle.load(f)
item_feature = get_ppr(top_k, item_idx_dict)
item_idx_tensor = item_feature.reshaped_tensor().to(device)
# number of items x (top_k x multi_factors)
del item_idx_dict

# user feature tensor
with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
    user_idx_dict = pickle.load(f)
user_feature = get_ppr(top_k, user_idx_dict)
user_idx_tensor = user_feature.reshaped_tensor().to(device)
del user_idx_dict

model = oneway_concat(item_idx_tensor,user_idx_tensor,
                      n_items, n_users, emb_dim, multi_factor, top_k).to(device)

print('\n')
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable parameters : ', pytorch_total_params)
print(next(model.parameters()).is_cuda)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
optimizer = optim.RMSprop(model.parameters(), lr=lr)
loss = nn.BCELoss()
print('optimizer : ', optimizer)
print('loss function : ', loss)


def train_sample(epoch, train_pos, train_neg, num_sample):
    print('\nepoch : ', epoch)
    model.train()
    optimizer.zero_grad()
    start = time.time()
    loss_list = []

    for user_idx in range(train_pos.shape[0]):
        # t1 = time.time()
        pos_row = train_pos.getrow(user_idx).toarray()[0]
        user_idxs = np.repeat(user_idx, num_sample)
        pos_items = np.where(pos_row == 1)[0]
        neg_row = train_neg.getrow(user_idx).toarray()[0]
        neg_items = np.where(neg_row == -1)[0]

        item_idxs = np.concatenate((pos_items, neg_items))
        concate_target = np.concatenate((np.ones_like(pos_items),
                                         np.zeros_like(neg_items)))
        targets = torch.Tensor(concate_target)
        predictions = model(user_idxs, item_idxs)

        train_loss = loss(predictions.to('cpu'), targets)
        loss_list.append(train_loss.detach().to('cpu').numpy())
        train_loss.backward()
        optimizer.step()
        # print('per user train takes : ', time.time() - t1)
    print('one epoch takes : ', time.time() - start)
    return loss_list


def train_all(epoch, train_all_mat):
    print('\nepoch : ', epoch)
    model.train()
    optimizer.zero_grad()
    start = time.time()
    loss_list = []

    for user_idx in range(train_all_mat.shape[0]):
        # t1 = time.time()
        user_row = train_all_mat.getrow(user_idx).toarray()[0]
        user_idxs = np.repeat(user_idx, n_items)
        item_idxs = list(range(n_items))
        targets = torch.Tensor(user_row)
        predictions = model(user_idxs, item_idxs)

        train_loss = loss(predictions.to('cpu'), targets)
        loss_list.append(train_loss.detach().to('cpu').numpy())
        train_loss.backward()
        optimizer.step()
        # print('per user train takes : ', time.time() - t1)
    print('one epoch takes : ', time.time() - start)
    return loss_list


def evaluate_sample(test_pos, test_neg, num_sample):
    model.eval()
    ndcg_20_list, recall_20_list = [], []

    t1 = time.time()
    for user_idx in range(test_pos.shape[0]):
        pos_row = test_pos.getrow(user_idx).toarray()[0]
        user_idxs = np.repeat(user_idx, num_sample)
        pos_items = np.where(pos_row == 1)[0]
        neg_row = test_neg.getrow(user_idx).toarray()[0]
        neg_items = np.where(neg_row == -1)[0]

        item_idxs = np.concatenate((pos_items, neg_items))
        # concate_targets = np.concatenate((np.ones_like(pos_items), np.zeros_like(neg_items)))
        # targets = torch.Tensor(concate_targets)
        # targets = torch.Tensor(np.ones_like(pos_items))
        predictions = model(user_idxs, item_idxs)

        recall_20_score = RECALL(predictions, pos_items, k=20)
        recall_20_list.append(recall_20_score)

        ndcg_20_score = NDCG(predictions, pos_items, k=20)
        ndcg_20_list.append(ndcg_20_score)
        if user_idx % 1000 == 0:
            print('up to user no. ' + str(user_idx) + ' processed!')
            t2 = time.time()
            print('takes : ', t2 - t1)
    return recall_20_list, ndcg_20_list


def evaluate_all(test_mat):
    model.eval()
    ndcg_20_list, recall_20_list = [], []

    t1 = time.time()
    for user_idx in range(test_mat.shape[0]):
        user_row = test_mat.getrow(user_idx).toarray()[0]
        user_idxs = np.repeat(user_idx, n_items)
        item_idxs = list(range(n_items))
        targets = user_row
        predictions = model(user_idxs, item_idxs)

        recall_20_score = RECALL_all(predictions, targets, k=20)
        recall_20_list.append(recall_20_score)

        ndcg_20_score = NDCG_all(predictions, targets, k=20)
        ndcg_20_list.append(ndcg_20_score)
        if user_idx % 1000 == 0:
            print('up to user no. ' + str(user_idx) + ' processed!')
            t2 = time.time()
            print('takes : ', t2 - t1)
    return recall_20_list, ndcg_20_list

# training loop
mean_epoch_loss = []
for epoch in range(epochs):
    loss_list = train_sample(epoch, train_pos, train_neg, num_sample)
    # loss_list = train_all(epoch, train_all_mat)
    print('mean epoch loss : ', np.mean(loss_list))
    mean_epoch_loss.append(np.mean(loss_list))

torch.save(model, os.path.join(data_dir,
                               'top_k_' + str(top_k)
                               + '_emb_' + str(emb_dim)
                               + '_num_sam_' + str(num_sample)
                               # + '_train_sam_' + str(num_sample)
                               # + '_test_sam_' + str(test_batch_size)
                               + '_epoch_' + str(epochs)
                               + '.model'))

# testing
print('\ntest started !')
t1 = time.time()
recall_20_list, ndcg_20_list = evaluate_sample(test_pos, test_neg, num_sample)
# recall_20_list, ndcg_20_list = evaluate_all(test_mat)
t2 = time.time()
print('evaluation takes : ', t2 - t1)

print('mean recall@20 : ', np.mean(recall_20_list))
print('mean NDCG@20 : ', np.mean(ndcg_20_list))

print('\nfinal user_emb min : ', torch.min(model.user_emb.weight))
print('final user_emb max : ', torch.max(model.user_emb.weight))
print('final user_emb mean : ', torch.mean(model.user_emb.weight))
print('final user_emb std : ', torch.std(model.user_emb.weight))

print('\nfinal item_emb min : ', torch.min(model.item_emb.weight))
print('final item_emb max : ', torch.max(model.item_emb.weight))
print('final item_emb mean : ', torch.mean(model.item_emb.weight))
print('final item_emb std : ', torch.std(model.item_emb.weight))
