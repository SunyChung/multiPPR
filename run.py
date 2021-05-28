import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model import ContextualizedNN
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/ml-1m/')
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--damping_factors', type=list, default=[0.30, 0.50, 0.70, 0.85, 0.95])
parser.add_argument('--learning_rate', type=float, default=0.005, help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=10, help='the number of epochs to train')
parser.add_argument('--input_dim', type=int, default=20, help='should be the top_k node idxs length')
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--output_dim', type=int, default=10)

args = parser.parse_args()
data_dir = args.data_dir
top_k = args.top_k
damping_factors = args.damping_factors
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = 1
input_dim = args.top_k  # 20
hidden_dim = input_dim * 10
output_dim = args.output_dim

with open(os.path.join(data_dir, 'item_cxt_top_20.dict'), 'rb') as f:
    item_cxt_dict = pickle.load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te = load_data(data_dir)
model = ContextualizedNN(data_dir, item_cxt_dict, input_dim, hidden_dim, output_dim, top_k).to(device)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable parameters : ', pytorch_total_params)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_coords, train_values, _ = get_sparse_coord_value_shape(train_data)
# print(train_coords)  # user, item index lists
# print(train_coords.shape)  # (480722, 2)
# print(train_values)

if device == 'cuda':
    train_data = train_data.to(device)
    vad_data_tr = vad_data_tr.to(device)
    vad_data_te = vad_data_te.to(device)
    test_data_tr = test_data_tr.to(device)
    test_data_te = test_data_te.to(device)
    model = model.to(device)


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
    # for batch_num, st_idx in enumerate(range(0, train_n, batch_size)):
    for batch_num, st_idx in enumerate(range(0, 110, batch_size)):
        print('batch num : ', batch_num)
        end_idx = min(st_idx + batch_size, 110)
        # end_idx = min(st_idx + batch_size, train_n)
        user_idxs = train_coords[idxlist[st_idx:end_idx]][:, 0]
        item_idxs = train_coords[idxlist[st_idx:end_idx]][:, 1]
        predictions = model(user_idxs, item_idxs)
        targets = torch.Tensor(train_values[idxlist[st_idx:end_idx]])
        # print('targets shape: ', targets.shape)  # torch.Size([100]) = batch_size
        train_loss = loss(predictions, targets)
        loss_list.append(train_loss.detach().numpy())
        train_loss.backward()
        optimizer.step()

    print('loss_list : ', loss_list)
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
        end_idx = min(st_idx+batch_size, test_n)
        user_idx = test_coords[idxlist[st_idx:end_idx]][:, 0]
        item_idx = test_coords[idxlist[st_idx:end_idx]][:, 1]
        prediction = model(user_idx, item_idx)
        targets = torch.Tensor(test_values[idxlist[st_idx:end_idx]])
        test_loss = loss(prediction, targets)
        loss_list.append(test_loss.detach().numpy())
    return loss_list


test_coords, test_values, _ = get_sparse_coord_value_shape(test_data_tr)
# print('test_coords ', test_coords)
# print('test_values ', test_values)
test_n = len(test_coords)
print('test_n : ', test_n)

loss_list = test(test_coords, test_values)
print(np.mean(loss_list))
