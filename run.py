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
parser.add_argument('--damping_factors', type=list, default=[0.30, 0.50, 0.70, 0.85, 0.95])
parser.add_argument('--learning_rate', type=float, default=5e-4, help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--multi_factor', type=int, default=5)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--emb_dim', type=int, default=100)
# parser.add_argument('--input_dim', type=int, default=20, help='top_k x multi_factor')
# parser.add_argument('--hidden_dim', type=int, default=100)
# parser.add_argument('--output_dim', type=int, default=10)
# parser.add_argument('--final_dim', type=int, default=1)

args = parser.parse_args()
data_dir = args.data_dir
damping_factors = args.damping_factors
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
multi_factor = args.multi_factor
top_k = args.top_k
emb_dim = args.emb_dim

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


def evaluate(test_input):
    model.eval()
    row, col = test_input.nonzero()
    uniq_users = np.unique(row)
    uniq_items = np.unique(col)
    input_array = test_input.toarray()
    loss = nn.BCELoss()
    loss_list = []
    recall_list = []
    ndcg_list = []
    for i in range(len(uniq_users)):
        target_user = uniq_users[i]
        target_user_idxs = np.where(input_array[target_user, :] == 1)[0]
        for j in range(len(uniq_items)):
            predictions = np.zeros_like(uniq_items)
            predictions[j] = model(target_user, uniq_items[j]).detach().cpu().numpy()
            recall_score = RECALL(predictions, target_user_idxs, k=20)
            recall_list.append(recall_score)
            ndcg_score = NDCG(predictions, target_user_idxs, k=20)
            ndcg_list.append(ndcg_score)
            test_loss = loss(predictions, target_user_idxs)
            loss_list.append(test_loss.detach())
    return recall_list, ndcg_list, loss_list


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
    recall_list, ndcg_list, loss_list = evaluate(valid_input)
    return recall_list, ndcg_list, loss_list


for epoch in range(epochs):
    recall_list, ndcg_list, loss_list = train(epoch, train_data, vad_data)
    print('returned recall :', np.mean(recall_list))
    print('returned NDCG : ', np.mean(ndcg_list))
    print('returned mean loss : ', np.mean(loss_list))


print('test started !')
recall_list, ndcg_list, loss_list = evaluate(test_data)
print('returned recall :', np.mean(recall_list))
print('returned NDCG : ', np.mean(ndcg_list))
print('returned mean loss : ', np.mean(loss_list))
