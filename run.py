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
parser.add_argument('--epochs', type=int, default=10)
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
epochs = 10
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
# print(train_coords)  # user, item index lists
# print(train_coords.shape)  # (480722, 2)
# print(train_values)


def train(epoch, train_coords, train_values):
    train_n = len(train_coords)
    # print(len(train_coords))  # 480722
    idxlist = list(range(train_n))
    print('epoch : ', epoch)
    start = time.time()
    model.train()
    optimizer.zero_grad()
    loss = nn.BCELoss()
    loss_list = []
    for batch_num, st_idx in enumerate(range(0, train_n, batch_size)):
        # print('batch num : ', batch_num)
        end_idx = min(st_idx + batch_size, train_n)
        user_idxs = train_coords[idxlist[st_idx:end_idx]][:, 0]
        item_idxs = train_coords[idxlist[st_idx:end_idx]][:, 1]
                                                          
        predictions = model(user_idxs, item_idxs)
        # reshaped_pred = torch.mean(predictions, dim=1).squeeze()
        targets = torch.Tensor(train_values[idxlist[st_idx:end_idx]])
        train_loss = loss(predictions.to('cpu'), targets)
        loss_list.append(train_loss.detach().item())
        train_loss.backward()
        optimizer.step()
    # print('loss_list : ', loss_list)
    print('epoch : {:04d}'.format(epoch),
          '\nmean train_loss : {:.4f}'.format(np.mean(loss_list)),
          '\ntime : {:.4f}s'.format(time.time() - start))


for epoch in range(epochs):
    train(epoch, train_coords, train_values)


print('test started !')


def test(test_coords, test_values):
    test_n = len(test_coords)
    idxlist = list(range(test_n))
    model.eval()
    loss = nn.BCELoss()
    loss_list = []
    for batch_num, st_idx in enumerate(range(0, test_n, batch_size)):
        print('batch_num : ', batch_num)
        end_idx = min(st_idx+batch_size, test_n)
        user_idx = test_coords[idxlist[st_idx:end_idx]][:, 0]
        item_idx = test_coords[idxlist[st_idx:end_idx]][:, 1]

        prediction = model(user_idx, item_idx)
        targets = torch.Tensor(test_values[idxlist[st_idx:end_idx]])
        test_loss = loss(prediction.to('cpu'), targets)
        loss_list.append(test_loss.detach())
    return loss_list


test_coords, test_values, _ = get_sparse_coord_value_shape(test_data_te)
test_n = len(test_coords)
print('test length : ', test_n)  # 18.853447

returned_loss = test(test_coords, test_values)
print(np.mean(returned_loss))
