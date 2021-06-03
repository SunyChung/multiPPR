import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from scipy.sparse import csr_matrix, isspmatrix_coo
from sknetwork.ranking import PageRank
import os
import time


def get_bipartite_matrix(data_dir):
    # "filtered_ratings.csv" 는 아직 index 값으로 mapping 안 된 상태 !!
    df = pd.read_csv(os.path.join(data_dir, 'filtered_ratings.csv'))

    with open(os.path.join(data_dir, 'uid_to_uidx.dict'), 'rb') as f:
        uid_to_uidx = pickle.load(f)
    with open(os.path.join(data_dir, 'mid_to_midx.dict'), 'rb') as f:
        mid_to_midx = pickle.load(f)
    with open(os.path.join(data_dir, 'train_mapped_id'), 'rb') as f:
        train_mapped_id = pickle.load(f)

    col = df['movieId'].map(mid_to_midx)  # Length: 574548,
    col = col.loc[col.isin(train_mapped_id)]  # Length: 574514,
    row = df['userId'].map(uid_to_uidx) # 574548
    row = row[col.index]  # Length: 574514
    # .map() or .apply(lambda x : )
    # row = df['userId'].apply(lambda x: uid_to_uidx[x])
    # col = df['movieId'].apply(lambda x: mid_to_midx[x])
    data = np.ones(len(row), dtype=int)
    # careful with the matirx shape !! : max() NOT len() -_;
    bi_matrix = csr_matrix((data, (row, col)), shape=(max(row)+1, max(col)+1))
    return bi_matrix


def get_movie_matrix(data_dir):
    bi_matrix = get_bipartite_matrix(data_dir)
    movie_matrix = bi_matrix.transpose() * bi_matrix
    return movie_matrix


def get_user_matrix(data_dir):
    bi_matrix = get_bipartite_matrix(data_dir)
    user_matrix = bi_matrix * bi_matrix.transpose()
    return user_matrix


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
        if i % 100 == 0:
            print('%d nodes processed!' % i)
            print('upto now %f seconds passed' %(time.time() - start))
    end = time.time()
    print('multi-ppr processing takes : ', end - start)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'wb') as f:
        pickle.dump(per_item_ppr_dict, f)
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'wb') as f:
        pickle.dump(per_item_idx_dict, f)

    _, per_user_item_dict = get_user_sequences(data_dir)
    with open(data_dir + 'per_user_item.dict', 'wb') as f:
        pickle.dump(per_user_item_dict, f)
    
   
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
        if i % 100 == 0:
            print('%d nodes processed!' % i)
            print('upto now %f seconds passed' % (time.time() - start))
    end = time.time()
    print('multi-ppr processing takes : ', end - start)

    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'wb') as f:
        pickle.dump(per_user_ppr_dict, f)
    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'wb') as f:
        pickle.dump(per_user_idx_dict, f)
    '''
