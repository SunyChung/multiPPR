import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
import os


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
    movie_matrix = bi_matrix.transpose() * bi_matrix
    return movie_matrix


def get_user_matrix(data_dir):
    bi_matrix = get_bipartite_matrix(data_dir)
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
