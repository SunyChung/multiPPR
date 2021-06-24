import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from scipy.sparse import csr_matrix, isspmatrix_coo
import os


def get_bipartite_matrix(data_dir):
    # "filtered_ratings.csv" 는 아직 index 값으로 mapping 안 된 상태 !!
    df = pd.read_csv(os.path.join(data_dir, 'filtered_ratings.csv'))

    with open(os.path.join(data_dir, 'train_sid'), 'rb') as f:
        train_sid = pickle.load(f)
    with open(os.path.join(data_dir, 'uid_to_uidx.dict'), 'rb') as f:
        uid_to_uidx = pickle.load(f)
    with open(os.path.join(data_dir, 'sid_to_sidx.dict'), 'rb') as f:
        sid_to_sidx = pickle.load(f)

    col = df['movieId']
    # print('col len : ', len(col))  # 574548
    col = col.loc[col.isin(train_sid)]
    # print('filtered col len : ', len(col))   # 574514
    # print('before id mapping col max : ', max(col))  # 3952 !!
    col = col.map(sid_to_sidx)  # 574514
    # print('mapped col len : ', len(col))  # 574514
    # print('after id mapping col max : ', max(col))  # 3504 !

    row = df['userId'].map(uid_to_uidx)
    # print('row len : ', len(row))  # 574548
    row = row[col.index]
    # print('row len : ', len(row))  # 574514

    data = np.ones(len(row), dtype=int)
    print('max(row) : ', max(row))  # 6030
    print('max(col) : ', max(col))  # 3504
    print('unique row : ', len(pd.unique(row)))  # 6031
    print('unique col : ', len(pd.unique(col)))  # 3505
    bi_matrix = csr_matrix((data, (row, col)), shape=(max(row) + 1, max(col) + 1))
    return bi_matrix


def get_movie_matrix(data_dir):
    bi_matrix = get_bipartite_matrix(data_dir)
    movie_matrix = bi_matrix.transpose() * bi_matrix
    return movie_matrix


def get_user_matrix(data_dir):
    bi_matrix = get_bipartite_matrix(data_dir)
    user_matrix = bi_matrix * bi_matrix.transpose()
    return user_matrix


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
    with open(os.path.join(data_dir, 'train_sid'), 'rb') as f:
        train_sid = pickle.load(f)
    # n_items = max(train_sid)  # 3952 이거 쓰면 안 됨!
    # mapping 된 값이 3505 이니, 이걸로 !!
    # sid_to_sidx 도 어차피 len(sid_to_sidx) = 3505
    n_items = len(train_sid)  # 3505

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


def get_all_from_sparse(sparse_mat):
    mat_array = sparse_mat.toarray()
    values = mat_array.flatten()
    coords = []
    for i in range(sparse_mat.shape[0]):
        for j in range(sparse_mat.shape[1]):
            coords.append(np.array([i, j]))
    return np.array(coords), values


def load_all(data_dir):
    with open(os.path.join(data_dir, 'train_sid'), 'rb') as f:
        train_sid = pickle.load(f)
    n_items = len(train_sid)
    train_sparse = load_train_data(os.path.join(data_dir, 'train.csv'), n_items)
    train_coords, train_values = get_all_from_sparse(train_sparse)
    vad_sparse = load_train_data(os.path.join(data_dir, 'vad.csv'), n_items)
    vad_coords, vad_values = get_all_from_sparse(vad_sparse)
    test_sparse = load_train_data(os.path.join(data_dir, 'test.csv'), n_items)
    test_coords, test_values = get_all_from_sparse(test_sparse)
    return n_items, train_coords, train_values, vad_coords, vad_values, test_coords, test_values


def get_sparse_coord_value_shape(sparse_mat):
    if not isspmatrix_coo(sparse_mat):
        sparse_mat = sparse_mat.tocoo()
    coords = np.vstack((sparse_mat.row, sparse_mat.col)).transpose()
    values = sparse_mat.data
    shape = sparse_mat.shape
    return coords, values, shape


def per_user_neg_idxs(data_dir, csv_file):
    with open(os.path.join(data_dir, 'train_sid'), 'rb') as f:
        train_sid = pickle.load(f)
    n_items = len(train_sid)
    
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    # unique_users = df['uid'].unique()
    user_item_dict = dict(df.groupby('uid')['sid'].apply(list))
    user_keys = np.array(list(user_item_dict.keys()))

    # df.groupby('uid')['sid] : returns `pandas.core.series.Series`
    # df.loc[df['uid']==6023]
    # len(user_item_dict[6023])
    user_neg_items = {}
    for user in user_keys:
        # should the values be list or array ?!
        neg = list(set(train_sid) - set(user_item_dict[0]))
        user_neg_items[user] = neg
    return user_neg_items
