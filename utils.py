import numpy as np
import pandas as pd
import pickle

from scipy import sparse
from scipy.sparse import csr_matrix, isspmatrix_coo
from sknetwork.ranking import PageRank
import os
import time
import bottleneck as bn


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


def load_vad_test_data(csv_file, n_items):
    df = pd.read_csv(csv_file)
    n_users = df['uid'].max() + 1
    rows, cols = df['uid'].to_numpy(), df['sid'].to_numpy()
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, n_items))
    return data


def load_all(data_dir):
    with open(os.path.join(data_dir, 'train_sid'), 'rb') as f:
        train_sid = pickle.load(f)
    n_items = len(train_sid)
    train_data = load_train_data(os.path.join(data_dir, 'train.csv'), n_items)
    vad_data = load_vad_test_data(os.path.join(data_dir, 'vad.csv'), n_items)
    test_data = load_vad_test_data(os.path.join(data_dir, 'test.csv'), n_items)
    return n_items, train_data, vad_data, test_data


def get_sparse_coord_value_shape(sparse_mat):
    if not isspmatrix_coo(sparse_mat):
        sparse_mat = sparse_mat.tocoo()
    coords = np.vstack((sparse_mat.row, sparse_mat.col)).transpose()
    values = sparse_mat.data
    shape = sparse_mat.shape
    return coords, values, shape


class MultiPPR(object):
    def __init__(self, damping_factors, matrix):
        super(MultiPPR, self).__init__()
        self.damping_factors = damping_factors
        self.mat = matrix

    def multi_contexts(self, target_idx):
        base_zeros = np.zeros(self.mat.shape[0] - 1)
        seed = np.insert(base_zeros, target_idx, 1)
        multi_score = []
        indices = []
        for i in range(len(self.damping_factors)):
            pagerank = PageRank(damping_factor=self.damping_factors[i])
            ppr = pagerank.fit_transform(self.mat.toarray(), seeds=seed)
            # don't need to exclude the first element from the start
            # can filtering the idx afterwards, if necessary
            idx = np.argsort(ppr)[::-1]
            sorted_scores = ppr[idx]
            multi_score.append(np.array(sorted_scores))
            indices.append(np.array(idx))
        return np.array(multi_score), np.array(indices)


def NDCG(pred_rel, item_idxs, k):
    # print('pred_rel : ', pred_rel)
    # 이거 다 1 만 찍고 있는데, 왜 argsort() 는 작은 값을 정렬하고 있는 건지 ?!
    topk_idx = np.argsort(pred_rel)[::-1][:k]
    topk_pred = pred_rel[topk_idx]
    # print('topk_pred : ', topk_pred)
    discount = 1. / np.log2(np.arange(2, k + 2))
    dcg = np.array([topk_pred / discount]).sum()
    idcg = np.array([discount[:min(len(item_idxs), k)]]).sum()
    # print('len(item_idx) : ', len(item_idxs))
    # print('idcg : ', idcg)
    return dcg / idcg


def RECALL(pred_rel, item_idx, k):
    topk_idx = np.argsort(pred_rel)[::-1][:k]
    pred_binary = np.zeros_like(pred_rel, dtype=bool)
    pred_binary[topk_idx] = True
    true_binary = np.zeros_like(pred_rel, dtype=bool)
    true_binary[item_idx] = True
    tmp = (np.logical_and(pred_binary, true_binary).sum()).astype(np.float32)
    recall = tmp / np.minimum(k, true_binary.sum())
    return recall


if __name__ == '__main__':
    # data_dir = './data/ml-1m/'
    data_dir = './data/ml-20m/'
    damping_factors = [0.30, 0.50, 0.70, 0.85, 0.95]

    movie_mat = get_movie_matrix(data_dir)
    print('movie_mat shape : ', movie_mat.shape)
    multi_ppr = MultiPPR(damping_factors, movie_mat)
    start = time.time()
    per_item_ppr_dict = {}
    per_item_idx_dict = {}
    for i in range(movie_mat.shape[0]):
        scores, indices = multi_ppr.multi_contexts(i)
        per_item_ppr_dict[i] = scores
        per_item_idx_dict[i] = indices
        if i % 1000 == 0:
            print('%d nodes processed!' % i)
            print('upto now %f seconds passed' % (time.time() - start))
    end = time.time()
    print('multi-ppr processing takes : ', end - start)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'wb') as f:
        pickle.dump(per_item_ppr_dict, f)
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'wb') as f:
        pickle.dump(per_item_idx_dict, f)


    '''
    user_mat = get_user_matrix(data_dir)
    print('user_mat shape : ', user_mat.shape)
    multi_ppr = MultiPPR(damping_factors, user_mat)

    start = time.time()
    # the default dictionary type is sufficient 'cause the return values already has multi array values !!!
    per_user_ppr_dict = {}
    per_user_idx_dict = {}
    for i in range(user_mat.shape[0]):
        scores, indices = multi_ppr.multi_contexts(i)
        per_user_ppr_dict[i] = scores
        per_user_idx_dict[i] = indices
        if i % 300 == 0:
            print('%d nodes processed!' % i)
            print('upto now %f seconds passed' % (time.time() - start))
    end = time.time()
    print('multi-ppr processing takes : ', end - start)

    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'wb') as f:
        pickle.dump(per_user_ppr_dict, f)
    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'wb') as f:
        pickle.dump(per_user_idx_dict, f)
    '''
