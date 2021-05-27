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
parser.add_argument('--epochs', type=int, default=10, help='the number of epochs to train')
parser.add_argument('--batch_size', type=int, default=100, help='the batch size for each epoch')
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--output_dim', type=int, default=10)

args = parser.parse_args()
data_dir = args.data_dir
top_k = args.top_k
damping_factors = args.damping_factors
learning_rate = args.learning_rate
epochs = 1
batch_size = args.batch_size
input_dim = top_k
hidden_dim = input_dim * 10
output_dim = args.output_dim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te = load_data(data_dir)
model = ContextualizedNN(data_dir, input_dim, hidden_dim, output_dim, top_k).to(device)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable parameters : ', pytorch_total_params)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_coords.to(device), train_values.to(device), _ = get_sparse_coord_value_shape(train_data)
# print(train_coords)
# print(train_coords.shape)  # (480722, 2)
# print(train_values)


def train(epoch, train_coords, train_values):
    # print(len(train_coords))  # 480722
    train_n = len(train_coords)
    idxlist = list(range(train_n))

    start = time.time()
    model.train()
    optimizer.zero_grad()
    loss = nn.BCELoss()
    loss_list = []
    # print('train_n : ', train_n)  # 480722
    # for batch_num, st_idx in enumerate(range(0, train_n, batch_size)):
    for batch_num, st_idx in enumerate(range(0, 110, batch_size)):
        print('batch num : ', batch_num)
        print('st_idx : ', st_idx)
        end_idx = min(st_idx + batch_size, 110)
        # end_idx = min(st_idx + batch_size, train_n)
        print('end_idx : ', end_idx)
        user_idxs = train_coords[idxlist[st_idx:end_idx]][:, 0]
        item_idxs = train_coords[idxlist[st_idx:end_idx]][:, 1]
        # print(user_idxs.shape)  # (100,)
        # print(item_idxs.shape)  # (100,)
        print('item_idxs ', item_idxs)
        predictions = model(user_idxs, item_idxs)
        print('predictions : ', predictions)
        # [tensor([0.5097], grad_fn=<SigmoidBackward>), tensor([0.5104], grad_fn=<SigmoidBackward>),
        # tensor([0.5083], grad_fn=<SigmoidBackward>), tensor([0.5086], grad_fn=<SigmoidBackward>), ... ]
        targets = torch.Tensor(train_values[idxlist[st_idx:end_idx]])
        print('targets : ', targets)
        train_loss = loss(predictions, targets)
        loss_list.append(train_loss.detach().numpy())
        # print(train_loss)
        train_loss.backward()
        optimizer.step()

    print(loss_list)
    print('epoch : {:04d}'.format(epoch),
          'train_loss : {:.4f}'.format(np.mean(loss_list)),
          'time : {:.4f}s'.format(time.time() - start))


for epoch in range(epochs):
    train(epoch, train_coords, train_values)


def test():
    model.eval()
    loss = nn.BCELoss()
    loss_list = []
    for i in range(test_n):
        user_idx = test_coords[i, 0]
        item_idx = test_coords[i, 1]
        prediction = model(user_idx, item_idx)
        target = torch.Tensor([test_values[i]])
        test_loss = loss(prediction, target)
        loss_list.append(test_loss)
    return loss_list


test_coords.to(device), test_values.to(device), _ = get_sparse_coord_value_shape(test_data_tr)
test_n = len(test_coords)

loss_list = test()
print(np.mean(loss_list))
