import os
import pickle
import time
import argparse
import numpy as np
import torch
from features import get_ppr
from load_data import Data
from utils import *


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
top_k = 20
num_sample = 10000

print('verification ...')
print('num of train samples : ', num_sample)

dataset = Data(data_dir)
# test_mat = dataset.make_test_all_mat()
n_users, n_items = dataset.get_num_users_items()
test_pos, test_neg = dataset.make_test_sample_mat(num_sample)

# item feature tensor
with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
    item_idx_dict = pickle.load(f)
item_feature = get_ppr(top_k, item_idx_dict)
item_idx_tensor = item_feature.reshaped_tensor()#.to(device)
# number of items x (top_k x multi_factors)
del item_idx_dict

# user feature tensor
with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
    user_idx_dict = pickle.load(f)
user_feature = get_ppr(top_k, user_idx_dict)
user_idx_tensor = user_feature.reshaped_tensor()#.to(device)
del user_idx_dict

print('CPU loading ...')
device = torch.device('cpu')
model = torch.load(os.path.join(data_dir,
    'top_k_20_emb_64_num_sam_10000_epoch_50.model')
    , map_location=device)
# model = torch.load(
#     'top_k_20_emb_32_epoch_50.model', map_location=device)
print(model)

# nohup_yelp_train_5000_test_all_emb_32_top_20_epoch_50
# top_k_20_emb_32_epoch_50.model

print('testing the saved model ... ')
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


recall_20_list, ndcg_20_list = evaluate_sample(test_pos, test_neg, num_sample)
# recall_20_list, ndcg_20_list = evaluate_all(test_mat)


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
