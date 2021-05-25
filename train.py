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
parser.add_argument('--batch_size', type=int, default=10, help='the batch size for each epoch')
parser.add_argument('--learning_rate', type=float, default=0.005, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10, help='the number of epochs to train')
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--output_dim', type=int, default=10)

args = parser.parse_args()
data_dir = args.data_dir
top_k = args.top_k
damping_factors = args.damping_factors
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
input_dim = top_k
hidden_dim = input_dim * 10
output_dim = args.output_dim

n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te = load_data(data_dir)

model = ContextualizedNN(data_dir, input_dim, hidden_dim, output_dim, top_k)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch):
    start = time.time()
    model.train()
    optimizer.zero_grad()
    loss = nn.BCELoss()
    loss_list = []
    # print('train_n : ', train_n)  # 480722
    for i in range(train_n):
        # index 는 굳이 Tensor 일 필요없음 !!
        user_idx = train_coords[i, 0]
        item_idx = train_coords[i, 1]
        # print(user_idx)
        # print(item_idx)
        prediction = model(user_idx, item_idx)
        # print('prediction : ', prediction.detach().cpu())
        # 그냥 다 1로 찍고 있 -_;;;
        # print(train_values[i])
        target = torch.Tensor([train_values[i]])
        train_loss = loss(prediction, target)
        loss_list.append(train_loss)
        # print(train_loss)
        train_loss.backward()
        optimizer.step()

    print('epoch : {:04d}'.format(epoch),
          'train_loss : {:.4f}'.format(np.mean(loss_list)),
          'time : {:.4f}s'.format(time.time() - start))


train_coords, train_values, _ = get_sparse_coord_value_shape(train_data)
print(train_coords)
print(train_coords.shape)
print(train_values)
train_n = train_coords.shape[0]
for epoch in range(epochs):
    train(epoch)


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


test_coords, test_values, _ = get_sparse_coord_value_shape(test_data_tr)
test_n = test_coords.shape[0]

loss_list = test()
