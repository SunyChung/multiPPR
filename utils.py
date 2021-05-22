import numpy as np
from scipy import sparse
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
import os
import torch


def get_bipartite_matrix(data_dir, n_users, n_items):
    df = pd.read_csv(data_dir + 'filtered_ratings.csv')
    with open(data_dir + 'uid_to_uidx.dict', 'rb') as f:
        uid_to_uidx = pickle.load(f)
    # 근데, movieId 는 train data 에 있는 것만 사용해야 되나 ??
    # indexing 이 애매하네...
    with open(data_dir + 'mid_to_midx.dict', 'rb') as f:
        mid_to_midx = pickle.load(f)
    row = df['userId'].apply(lambda x: uid_to_uidx[x])
    col = df['movieId'].apply(lambda x: mid_to_midx[x])
    # row = np.array(df['userId']) - 1
    # col = np.array(df['movieId']) - 1
    data = np.ones(len(row) * len(col), dtype=int)
    bi_matrix = csr_matrix((data, (row, col)), shape=(n_users, n_items))
    return bi_matrix


def get_movie_matrix(data_dir):
    bi_matrix = get_bipartite_matrix(data_dir)
    movie_matrix = bi_matrix.transpose() * bi_matrix
    return movie_matrix


def get_user_sequences(data_dir):
    df = pd.read_csv(data_dir + 'filtered_ratings.csv')
    df_array = df.to_numpy()
    user_array = np.unique(df_array[:, 0])
    per_user_item_dict = {}
    for i in user_array:
        per_user_item_dict[i] = df_array[np.where(df_array[:, 0] == i)][:, 1]
    return user_array, per_user_item_dict
    # print(max(per_user_item_indices))  # 1435
    # print(min(per_user_item_indices))  # 1
    # print(np.mean(per_user_item_indices))  # 95.26579340076273


def load_train_data(csv_file, n_items):
    df = pd.read_csv(csv_file)
    n_users = df['uid'].max() + 1
    rows, cols = df['uid'].to_numpy(), df['sid'].to_numpy()
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    df_tr = pd.read_csv(csv_file_tr)
    df_te = pd.read_csv(csv_file_te)

    start_idx = min(df_tr['uid'].min(), df_te['uid'].min())
    end_idx = max(df_tr['uid'].max(), df_te['uid'].max())

    rows_tr, cols_tr = df_tr['uid'] - start_idx, df_tr['sid']
    rows_te, cols_te = df_te['uid'] - start_idx, df_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)),
                                dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)),
                                dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def load_data(data_dir):
    with open(os.path.join(data_dir, 'unique_sid'), 'rb') as f:
        unique_sid = pickle.load(f)
    n_items = len(unique_sid)  # + 1 ??

    train_data = load_train_data(os.path.join(data_dir, 'train.csv'), n_items)
    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(data_dir, 'vad_tr.csv'),
                                               os.path.join(data_dir, 'vad_te.csv'),
                                               n_items)
    test_data_tr, test_data_te = load_tr_te_data(os.path.join(data_dir, 'test_tr.csv'),
                                                 os.path.join(data_dir, 'test_te.csv'),
                                                 n_items)

    assert n_items == train_data.shape[1]
    assert n_items == vad_data_tr.shape[1]
    assert n_items == vad_data_te.shape[1]
    assert n_items == test_data_tr.shape[1]
    assert n_items == test_data_te.shape[1]
    return n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te
    return generator


def sparse2tensor(data):
    rows = data.shape[0]
    cols = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.rows, coo_data.cols])
    # 근데, normalize 해야 되나 ?????
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i: row_norms_inv[i].item() for i in range(rows)}
    values = np.array([row2val[r]] for r in coo_data.rows)
    # torch.Tensor is an alias for the default tensor type (torch.FloatTensor)
    t = torch.FloatTensor(indices, torch.from_numpy(values).float(), [rows, cols])
