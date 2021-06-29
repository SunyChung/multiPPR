import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix, isspmatrix_coo
from collections import defaultdict
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
    print('max(col) : ', max(col))  # 3502  # 3504
    print('unique row : ', len(pd.unique(row)))  # 6031
    print('unique col : ', len(pd.unique(col)))  # 3503 # 3505
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
def get_target_1(csv_file):
    df = pd.read_csv(csv_file)
    triplet = []
    for i in range(len(df)):
        triplet.append(np.array([df['uid'][i], df['sid'][i], 1]))
    return np.array(triplet)


def get_target_0(csv_file, train_sid):
    df = pd.read_csv(csv_file)
    group_dict = df.groupby('uid').groups
    triplet = []
    for i in group_dict.keys():
        target_0_items = list(set(train_sid) - set(group_dict[i]))
        for j in range(len(target_0_items)):
            triplet.append(np.array([i, target_0_items[j], 0]))
    return np.array(triplet)


# need to shuffle in user group!
def get_targets(csv_file, train_sid):
    df = pd.read_csv(csv_file)
    group_dict = df.groupby('uid').groups
    triplet = []
    for i in group_dict.keys():
        for j in group_dict[i]:
            triplet.append(np.array([i, j, 1]))
        target_0_items = list(set(train_sid) - set(group_dict[i]))
        for k in range(len(target_0_items)):
            triplet.append(np.array([i, target_0_items[k], 0]))
    return np.array(triplet)  # (18093347, 3)


def triplet_dict(csv_file, train_sid):
    df = pd.read_csv(csv_file)
    grouped_df = df.groupby('uid').sid.apply(np.array).reset_index()
    uid_sid = defaultdict(list)
    for i in range(len(grouped_df)):
        for j in grouped_df['sid'][i]:
            uid_sid[grouped_df['uid'][i]].append(np.array([j, 1]))
        target_0_items = list(set(train_sid) - set(grouped_df['sid'][i]))
        for k in target_0_items:
            uid_sid[grouped_df['uid'][i]].append(np.array([k, 0]))
    return uid_sid


def dict_shuffle(dic):
    shuffled = {}
    for i in dic.keys():
        shuffled = dic[i].shuffle
    return shuffled


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
