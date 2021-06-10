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

'''
# user 를 items sequence 로 사용하지 않아서 이거 없어도 됨 !
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
'''


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


def NDCG(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def RECALL(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)

    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


if __name__ == '__main__':
    data_dir = './data/ml-1m/'
    damping_factors = [0.30, 0.50, 0.70, 0.85, 0.95]

    # PPR calculation takes about 5 hours on my Mac (16GB),
    # and takes about 3 hours on desktop (32GB)
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
        if i % 300 == 0:
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
