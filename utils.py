import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from scipy.sparse import csr_matrix, isspmatrix_coo
import os
import torch


def get_bipartite_matrix(data_dir):
    # "filtered_ratings.csv" 는 아직 index 값으로 mapping 안 된 상태 !!
    df = pd.read_csv(os.path.join(data_dir, 'filtered_ratings.csv'))
    # user, movie indexing 을 위해 dictionary 읽어오고
    with open(os.path.join(data_dir, 'uid_to_uidx.dict'), 'rb') as f:
        uid_to_uidx = pickle.load(f)
    with open(os.path.join(data_dir, 'mid_to_midx.dict'), 'rb') as f:
        mid_to_midx = pickle.load(f)
    # movieId 의 경우, train dataset 에 포함된 Id 만 선택 !!
    with open(os.path.join(data_dir, 'train_mapped_id'), 'rb') as f:
        train_mapped_id = pickle.load(f)

    row = df['userId'].map(uid_to_uidx)
    col = df['movieId'].map(mid_to_midx)  # Length: 574548,
    col = col.loc[col.isin(train_mapped_id)]  # Length: 574514,
    # .map() or .apply(lambda x : )
    # row = df['userId'].apply(lambda x: uid_to_uidx[x])
    # col = df['movieId'].apply(lambda x: mid_to_midx[x])
    data = np.ones(len(row), dtype=int)
    bi_matrix = csr_matrix((data, (row, col)), shape=(len(uid_to_uidx), len(train_mapped_id)))
    return bi_matrix


def get_movie_matrix(data_dir):
    bi_matrix = get_bipartite_matrix(data_dir)
    movie_matrix = bi_matrix.transpose() * bi_matrix
    return movie_matrix


# 이것도 index 로 mapping 시켜야 되나 ?? YES !!!
def get_user_sequences(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'filtered_ratings.csv'))
    with open(os.path.join(data_dir, 'uid_to_uidx.dict'), 'rb') as f:
        uid_to_uidx = pickle.load(f)
    with open(os.path.join(data_dir, 'mid_to_midx.dict'), 'rb') as f:
        mid_to_midx = pickle.load(f)

    df['userId'] = df['userId'].map(uid_to_uidx)
    df['movieId'] = df['movieId'].map(mid_to_midx)
    df_array = df.to_numpy()
    user_array = df['userId'].unique()
    per_user_item_dict = {}
    for i in user_array:
        per_user_item_dict[i] = df_array[np.where(df_array[:, 0] == i)][:, 1]
    return user_array, per_user_item_dict
    # print(max(per_user_item_indices))  # 1435
    # print(min(per_user_item_indices))  # 1
    # print(np.mean(per_user_item_indices))  # 95.26579340076273


# train, validation, test data 는 모두 index 값으로 mapping 된 상태 !
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
    # 기존 코드들이 사용하던 unique_sid 가 아니라, train_mapped_id 써야 됨 !!
    with open(os.path.join(data_dir, 'train_mapped_id'), 'rb') as f:
        train_mapped_id = pickle.load(f)
    # n_items = len(train_mapped_id)  # + 1 ??
    # length 가 아니라, id max 값 써야 함 !!
    n_items = max(train_mapped_id) + 1

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


def get_sparse_coord_value_shape(sparse_mat):
    if not isspmatrix_coo(sparse_mat):
        sparse_mat = sparse_mat.tocoo()
    coords = np.vstack((sparse_mat.row, sparse_mat.col)).transpose()
    values = sparse_mat.data
    shape = sparse_mat.shape
    return coords, values, shape
