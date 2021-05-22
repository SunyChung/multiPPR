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
parser.add_argument('--batch_size', type=int, default=64, help='the batch size for each epoch')
parser.add_argument('--learning_rate', type=float, default=0.005, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10, help='the number of epochs to train')
parser.add_argument('--hidden_dim', type=int, default=40)
parser.add_argument('--output_dim', type=int, default=10)

args = parser.parse_args()
data_dir = args.data_dir
top_k = args.top_k
damping_factors = args.damping_factors
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
input_dim = top_k
hidden_dim = args.hidden_dim
output_dim = args.output_dim


bipartite_mat = get_bipartite_matrix(data_dir)
n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te = load_data(data_dir)

model = ContextualizedNN(data_dir, input_dim, hidden_dim, output_dim, top_k)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

N = train_data.shape[0]
idxlist = list(range(N))
def train(epoch, user_idx, item_idx):
    model.train()
    start = time.time()

    # batch 로 data index 넘겨서 model 에서 batch 로 반환받는 방법 찾을 것 !!
    for batch_idx, start_idx in enumerate(range(0, N, batch_size)):
        end_idx = min(start_idx+batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = torch.Tensor(data.toarray())

    optimizer.zero_grad()
    loss = nn.BCELoss()  # binary cross entropy loss
    output = model(user_idx, item_idx)

    # target 은 st_idx, end_idx 로 값 불러오면 되는데,
    # model 에서 batch 로 반환 받으려면 ???
    target = torch.Tensor([ bipartite_mat[user_idx, item_idx] ])

    train_loss = loss(output, target)
    train_loss.backward()
    optimizer.step()

    # validation loss & accuracy should be added
    # accuracy method will be implemented in `utils.py`
    # val_loss =
    print('epoch : {:04d}'.format(epoch),
          'train_loss : {:.4f}'.format(train_loss.data.item()),
          'time : {:.4f}s'.format(time.time() - start)
          )
    return train_loss


# tr_row, tr_col should be split into batches ...!
tr_row, tr_col = train_data.nonzero()


for epoch in range(epochs):
    for i in range(len(tr_row)):
        train(epoch, tr_row[i], tr_col[i])
