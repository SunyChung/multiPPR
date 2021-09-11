import os
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from features import PPRfeatures
from utils import load_all

from twoway_concat_model import twoway_concat

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

epochs = 50
multi_factor = args.multi_factor
top_k = 20
emb_dim = 64
print('learning rate : ', lr)
print('multi factors : ', multi_factor)
print('epochs : ', epochs)
print('top_k : ', top_k)
print('embedding dimension : ', emb_dim)

n_items, train_data, vad_data, test_data = load_all(data_dir)
print('n_items : ', n_items)
# batch_size = 500
# batch size 가 n_items 가 아니면, NDCG 및 RECALL 이 inf, nan 반환 -_
batch_size = n_items
print('batch size : ', batch_size)
with open(os.path.join(data_dir, 'unique_uidx'), 'rb') as f:
    unique_uidx = pickle.load(f)
n_users = len(unique_uidx)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using {} device'.format(device))

# item feature tensor
with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
    item_idx_dict = pickle.load(f)
with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
    item_ppr_dict = pickle.load(f)
item_pf = PPRfeatures(top_k, item_idx_dict, item_ppr_dict)
item_idx_tensor = item_pf.reshaped_idx_tensor().to(device)
item_scr_tensor = item_pf.reshaped_scr_tensor().to(device)

del item_idx_dict
del item_ppr_dict

# user feature tensor
with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
    user_idx_dict = pickle.load(f)
with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
    user_ppr_dict = pickle.load(f)
user_pf = PPRfeatures(top_k, user_idx_dict, user_ppr_dict)
user_idx_tensor = user_pf.reshaped_idx_tensor().to(device)
user_scr_tensor = user_pf.reshaped_scr_tensor().to(device)

del user_idx_dict
del user_ppr_dict

model = twoway_concat(item_idx_tensor, item_scr_tensor,
                      user_idx_tensor, user_scr_tensor,
                      data_dir, n_items, n_users, emb_dim, multi_factor, top_k).to(device)

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


def train(epoch, train_data, vad_data):
    print('\nepoch : ', epoch)
    start = time.time()
    train_n = len(train_data)
    train_idxlist = list(range(train_n))
    model.train()
    optimizer.zero_grad()

    loss_list = []
    for batch_num, st_idx in enumerate(range(0, train_n, batch_size)):
        end_idx = min(st_idx + batch_size, train_n)
        user_idxs = train_data[train_idxlist[st_idx:end_idx]][:, 0]
        item_idxs = train_data[train_idxlist[st_idx:end_idx]][:, 1]
        # (3515,)
        targets = torch.Tensor(train_data[train_idxlist[st_idx:end_idx]][:, 2])
        # tensor([0., 0., 0.,  ..., 0., 0., 0.])
        # torch.Size([3515])
        predictions = model(user_idxs, item_idxs)
        train_loss = loss(predictions.to('cpu'), targets)
        loss_list.append(train_loss.detach().to('cpu').numpy())
        train_loss.backward()
        optimizer.step()
    print('one epoch takes : ', time.time() - start)
    ndcg_100_list, recall_100_list, recall_50_list, recall_20_list \
        = evaluate(vad_data)
    return ndcg_100_list, recall_100_list, recall_50_list, recall_20_list, loss_list


def NDCG(predictions, targets, k):
    topk_idx = np.argsort(predictions)[::-1][:k]
    discount = 1. / np.log2(np.arange(2, k+2))
    DCG = np.array(targets[topk_idx] * discount).sum()
    IDCG = discount[:min(targets.sum(), k)].sum()
    return DCG / IDCG


def RECALL(predictions, targets, k):
    topk_idx = np.argsort(predictions)[::-1][:k]
    pred_binary = np.zeros_like(predictions, dtype=bool)
    pred_binary[topk_idx] = True
    true_binary = targets > 0
    tmp = (np.logical_and(pred_binary, true_binary).sum()).astype(np.float32)
    dinorm = min(k, targets.sum())
    return tmp / dinorm


def evaluate(test_data):
    model.eval()
    # print('evaluation batch length : ', len(test_data))
    test_n = len(test_data)
    test_idxlist = list(range(test_n))
    ndcg_100_list, recall_100_list, recall_50_list, recall_20_list\
        = [], [], [], []

    for batch_num, st_idx in enumerate(range(0, test_n, batch_size)):
        end_idx = min(st_idx + batch_size, test_n)
        user_idxs = test_data[test_idxlist[st_idx:end_idx]][:, 0]
        item_idxs = test_data[test_idxlist[st_idx:end_idx]][:, 1]
        predictions = model(user_idxs, item_idxs).to('cpu').detach().numpy()
        targets = test_data[test_idxlist[st_idx:end_idx]][:, 2]

        ndcg_100_score = NDCG(predictions, targets, k=100)
        ndcg_100_list.append(ndcg_100_score)

        recall_100_score = RECALL(predictions, targets, k=100)
        recall_100_list.append(recall_100_score)

        recall_50_score = RECALL(predictions, targets, k=50)
        recall_50_list.append(recall_50_score)

        recall_20_score = RECALL(predictions, targets, k=20)
        recall_20_list.append(recall_20_score)
    return ndcg_100_list, recall_100_list, recall_50_list, recall_20_list

# training loop
mean_epoch_loss, mean_ndcg_100, \
mean_recall_100, mean_recall_50, mean_recall_20 \
    = [], [], [], [], []
# std_epoch_loss, std_ndcg_100, std_recall_100, std_recall_50, std_recall_20 = [], [], [], [], []
for epoch in range(epochs):
    ndcg_100_list, recall_100_list, recall_50_list, recall_20_list, loss_list \
        = train(epoch, train_data, vad_data)
    print('\ntraining evaluation ... ')
    print('mean epoch loss : ', np.mean(loss_list))
    mean_epoch_loss.append(np.mean(loss_list))
    print('mean NDCG@100 : ', np.mean(ndcg_100_list))
    mean_ndcg_100.append(np.mean(ndcg_100_list))
    print('mean RECALL@100 : ', np.mean(recall_100_list))
    mean_recall_100.append(np.mean(recall_100_list))
    print('mean RECALL@50 : ', np.mean(recall_50_list))
    mean_recall_50.append(np.mean(recall_50_list))
    print('mean RECALL@20 : ', np.mean(recall_20_list))
    mean_recall_20.append(np.mean(recall_20_list))
torch.save(model, os.path.join(data_dir, 'twoway_concat.model'))

# testing
print('\ntest started !')
ndcg_100_list, recall_100_list, recall_50_list, recall_20_list = evaluate(test_data)
print('mean NDCG@100 : ', np.mean(ndcg_100_list))
print('mean RECALL@100 : ', np.mean(recall_100_list))
print('mean RECALL@50 : ', np.mean(recall_50_list))
print('mean RECALL@20 : ', np.mean(recall_20_list))

out_dir = './figures/' \
          + 'two_concat/' \
          + 'epo_' + str(epochs) + '_top_' + str(top_k) \
          + '_emb_' + str(emb_dim) \
          + '_loss_BCE' + '_optim_RMSprop' + '_default_init_/'
# + '_loss_BCE' + '_optim_RMSprop' + '_kaiming_normal_/'
# + '_loss_BCE' + '_optim_RMSprop' + '_xavier_normal_/'

# out_dir = './ex/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def result_plot(epochs, results, plot_label, y_label, save_name, title_label):
    params = {'legend.fontsize': 'small',
              'figure.figsize': (3.5, 2.5),
              'axes.labelsize': 'x-small',
              'axes.titlesize': 'x-small',
              'xtick.labelsize': 'xx-small',
              'ytick.labelsize': 'xx-small'}
    plt.rcParams['figure.constrained_layout.use'] = True
    pylab.rcParams.update(params)
    plt.figure()
    plt.plot(epochs, results, label=plot_label)
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    plt.title(title_label)
    plt.savefig(save_name)
    plt.show()


epoch_range = range(1, epochs + 1)
# print('mean_epoch_loss', mean_epoch_loss)
result_plot(np.array(epoch_range), np.array(mean_epoch_loss),
            plot_label='mean epoch loss', y_label='mean loss',
            save_name=out_dir + 'mean_loss.png',
            title_label= 'two_concat_'
                        + '_top_' + str(top_k) + '_emb_' + str(emb_dim)
                        + '_loss_BCE' + '_opt_RMS' + '_weight_default')
# + '_loss_BCE' + '_opt_RMS' + '_kaiming')
# + '_loss_BCE' + '_opt_RMS' + '_xavier')

# print('mean_ndcg_100', mean_ndcg_100)
result_plot(np.array(epoch_range), np.array(mean_ndcg_100),
            plot_label='mean NDCG@100', y_label='NDCE@100',
            save_name=out_dir + 'mean_NDCE_100.png',
            title_label= 'two_concat_'
                        + '_top_' + str(top_k) + '_emb_' + str(emb_dim)
                        + '_loss_BCE' + '_opt_RMS' + '_weight_default')

# print('mean_recall_100', mean_recall_100)
result_plot(np.array(epoch_range), np.array(mean_recall_100),
            plot_label='mean recall@100', y_label='RECALL@100',
            save_name=out_dir + 'mean_recall_100.png',
            title_label= 'two_concat_'
                        + '_top_' + str(top_k) + '_emb_' + str(emb_dim)
                        + '_loss_BCE' + '_opt_RMS' + '_weight_default')

# print('mean_recall_50', mean_recall_50)
result_plot(np.array(epoch_range), np.array(mean_recall_50),
            plot_label='mean recall@50', y_label='RECALL@50',
            save_name=out_dir + 'mean_recall_50.png',
            title_label= 'two_concat_'
                        + '_top_' + str(top_k) + '_emb_' + str(emb_dim)
                        + '_loss_BCE' + '_opt_RMS' + '_weight_default')

# print('mean_recall_20', mean_recall_20)
result_plot(np.array(epoch_range), np.array(mean_recall_20),
            plot_label='mean recall@20', y_label='RECALL@20',
            save_name=out_dir + 'mean_recall_20.png',
            title_label= 'two_concat_'
                        + '_top_' + str(top_k) + '_emb_' + str(emb_dim)
                        + '_loss_BCE' + '_opt_RMS' + '_weight_default')

