import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
import os
import time
from collections import Counter
from collections import defaultdict


def get_bipartite_matrix(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'filtered_pd.csv'))
    # load filtered & idx mapped dataframe !
    col = df['movieId']
    row = df['userId']
    data = np.ones(len(row), dtype=int)
    print('bipartite mat : max(row) = ', max(row))
    print('bipartite mat : max(col) = ', max(col))
    print('unique # of row : ', len(pd.unique(row)))
    print('unique # of col : ', len(pd.unique(col)))
    bi_matrix = csr_matrix((data, (row, col)), shape=(max(row)+1, max(col)+1))
    return bi_matrix


def get_movie_matrix(data_dir):
    bi_matrix = get_bipartite_matrix(data_dir)
    # weighted
    movie_matrix = bi_matrix.transpose() * bi_matrix
    '''
    row, col = movie_matrix.nonzero()
    movie_adj = csr_matrix((np.ones_like(row), (row, col)),
                           shape=(max(row)+1, max(col)+1))
    # 굳이 adj 로 바꿀 필요 없음 ... ;
    return movie_adj
    '''
    return movie_matrix


def get_user_matrix(data_dir):
    bi_matrix = get_bipartite_matrix(data_dir)
    # weighted
    user_matrix = bi_matrix * bi_matrix.transpose()
    return user_matrix


def get_pos_neg_df(csv_file, train_sid):
    df = pd.read_csv(csv_file)
    df['target'] = 1

    user_negatives = []
    for i in df['userId'].unique():
        movie_array = df['movieId'][df['userId'] == i].to_numpy()
        target_0 = list(set(train_sid) - set(movie_array))
        user_negatives.append(np.array([np.repeat(i, len(target_0)), target_0]).T)
    pd_list = []
    for i in range(len(user_negatives)):
        pd_list.append(pd.DataFrame(user_negatives[i], columns=['userId', 'movieId']))
    n_df = pd.concat(pd_list)
    n_df['target'] = 0
    merged = pd.concat([df, n_df])
    shuffled_pd = merged.groupby('userId').sample(frac=1)
    return shuffled_pd


def load_all(data_dir):
    with open(os.path.join(data_dir, 'train_sid'), 'rb') as f:
        train_sid = pickle.load(f)
    n_items = len(train_sid)

    train_df = get_pos_neg_df(os.path.join(data_dir, 'train.csv'), train_sid)
    train_np = train_df.to_numpy()
    vad_df = get_pos_neg_df(os.path.join(data_dir, 'vad.csv'), train_sid)
    vad_np = vad_df.to_numpy()
    test_df = get_pos_neg_df(os.path.join(data_dir, 'test.csv'), train_sid)
    test_np = test_df.to_numpy()
    return n_items, train_np, vad_np, test_np


def make_user_neigh_dict(df):
    user_item_dict = df.groupby('userId')['movieId'].apply(list).to_dict()
    item_user_dict = df.groupby('movieId')['userId'].apply(list).to_dict()

    per_user_neigh = defaultdict(list)
    for user in list(user_item_dict.keys()):
        item_seq = user_item_dict[user]
        for item in item_seq:
            user_seq = item_user_dict[item]
            per_user_neigh[user].extend(user_seq)

    per_user_sorted_neigh = defaultdict(list)
    for user in list(per_user_neigh.keys()):
        sorted_neigh = [neigh for neigh, cnt in Counter(per_user_neigh[user]).most_common()]
        per_user_sorted_neigh[user] = sorted_neigh

    del per_user_neigh
    return per_user_sorted_neigh


if __name__ == '__main__':
    # data_dir = './data/ml-1m'
    # 정작 필요한 ml-20m 에서는 SIGKILL 로 쓰지도 못하고 ...
    data_dir = './data/ml-20m'

    start = time.time()
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    per_user_sorted_neigh = make_user_neigh_dict(df)
    end = time.time()
    print('takes : ', end - start)
    with open(os.path.join(data_dir, 'per_user_neigh.dict'), 'wb') as f:
        pickle.dump(per_user_sorted_neigh, f)

    print('len(per_user_sorted_neigh) : ', len(per_user_sorted_neigh))  # 5034
    # df['userId'].unique()
    # array([   0,    1,    2, ..., 6029, 6031, 6033])
    # len(df['userId'].unique())
    # 5034
    # per_user_sorted_neigh 길이 확인 !
    neigh_length = []
    for i in list(per_user_sorted_neigh.keys()):
        neigh_length.append(len(per_user_sorted_neigh[i]))
    print('mean neighbor length : ', np.mean(neigh_length))
    print('min neighbor length : ', np.min(neigh_length))
    print('max neighbor length : ', np.max(neigh_length))

    # 아래 상황은, per_user_sorted_neigh.keys() 를 사용하지 않고,
    # range(len(per_user_sorted_neigh) 로 하다 발생한 문제인데,
    # 근데, 왜 문제가 생겼지 ?! -_;
    # 아, range() 를 쓰면, key 값이 0 부터 순차적으로 대입되는데,
    # 사실 userId 가 중간에 filtering 으로 빠지는 부분도 존재하기 때문에
    # 없는 userId 도 포함됨

    # neighbor 수가 0 인 경우가 많음 ... 무려 966 명
    # 근데, neighbor 가 0 이라는 건, 해당 user 의 item 이 하나도 없다는 건데,
    # 이미 data filtering 에서 이거 걸렀을텐데 ?!?!
    neigh_length_0 = []
    for i, l in enumerate(neigh_length):
        if l == 0:
            neigh_length_0.append(i)
    print('len(neigh_length_0) : ', len(neigh_length_0))
