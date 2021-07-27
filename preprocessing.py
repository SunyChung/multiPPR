import pandas as pd
import numpy as np
import pickle
import sys
import os


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_sc=0, min_uc=5):
    if min_sc > 0:
        item_count = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(item_count['movieId'][item_count['size'] >= min_sc].to_numpy())]
        print('movieId filtered !')

    if min_uc > 0:
        user_count = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(user_count['userId'][user_count['size'] >= min_uc].to_numpy())]
        # print(tp)
        print('userId filtered!')
    user_count, item_count = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, user_count, item_count


def data_filtering(data_dir, tp):
    tp, user_activity, item_popularity = filter_triplets(tp)
    sparsity = 1. * tp.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    print('after filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)'
          % (tp.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))
    # ml-1m
    # after filtering, there are 575272 watching events
    # from 6034 users and 3533 movies (sparsity: 2.699%)
    # ml-20m
    # after filtering, there are 9990682 watching events
    # from 136677 users and 20720 movies (sparsity: 0.353%)
    tp = tp[['userId', 'movieId']]
    with open(os.path.join(data_dir, 'filtered_ratings.csv'), 'w') as f:
        tp.to_csv(f, index=False)
    return tp, user_activity, item_popularity


if __name__ == '__main__':
    data_dir = './data/ml-1m'
    # data_dir = './data/ml-20m'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    threshold = 3.5
    n_held_out_users = 500  # for ml-1m
    # n_held_out_users = 10000  # for ml-20m

    test_prop = 0.2
    raw_data = pd.read_csv(os.path.join(data_dir, 'ratings.dat'), sep='::',
                           names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
    # raw_data = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))

    raw_data = raw_data[raw_data['rating'] > threshold]
    filtered_pd, user_count, item_count = data_filtering(data_dir, raw_data)

    unique_uidx = user_count.index
    print('number of total users : ', len(unique_uidx))
    with open(os.path.join(data_dir, 'unique_uidx'), 'wb') as f:
        pickle.dump(unique_uidx, f)

    uidx_to_uid = user_count['userId'].to_dict()
    with open(os.path.join(data_dir, 'uidx_to_uid.dict'), 'wb') as f:
        pickle.dump(uidx_to_uid, f)

    uid_to_uidx = {value: key for (key, value) in uidx_to_uid.items()}
    with open(os.path.join(data_dir, 'uid_to_uidx.dict'), 'wb') as f:
        pickle.dump(uid_to_uidx, f)

    np.random.seed(1234)
    idx_perm = np.random.permutation(unique_uidx.size)
    unique_uidx = unique_uidx[idx_perm]

    n_users = unique_uidx.size
    tr_users = unique_uidx[:(n_users - n_held_out_users * 2)]
    vd_users = unique_uidx[(n_users - n_held_out_users * 2):(n_users - n_held_out_users)]
    te_users = unique_uidx[(n_users - n_held_out_users):]
    print('# of total users : ', n_users)
    print('# of train users : ', len(tr_users))
    print('# of validation users : ', len(vd_users))
    print('# of test users : ', len(te_users))

    filtered_pd['userId'] = filtered_pd['userId'].map(uid_to_uidx)
    train_plays = filtered_pd.loc[filtered_pd['userId'].isin(tr_users)]
    train_movie = pd.unique(train_plays['movieId'])
    print('length of unique train items : ', len(train_movie))
    # filtering movieId with train movies
    filtered_pd = filtered_pd.loc[filtered_pd['movieId'].isin(train_movie)]

    # sid (1 부터 시작 !) 와 sidx (0 부터 시작 !)
    sid_to_sidx = {sid: i for (i, sid) in enumerate(train_movie)}
    with open(os.path.join(data_dir, 'sid_to_sidx.dict'), 'wb') as f:
        pickle.dump(sid_to_sidx, f)

    sidx_to_sid = {value: key for (key, value) in sid_to_sidx.items()}
    with open(os.path.join(data_dir, 'sidx_to_sid.dict'), 'wb') as f:
        pickle.dump(sidx_to_sid, f)

    train_sid = np.array([sid_to_sidx[x] for x in train_movie])
    with open(os.path.join(data_dir, 'train_sid'), 'wb') as f:
        pickle.dump(train_sid, f)
    filtered_pd['movieId'] = filtered_pd['movieId'].map(sid_to_sidx)
    filtered_pd.to_csv(os.path.join(data_dir, 'filtered_pd.csv'), index=False)

    train_plays['movieId'] = train_plays['movieId'].map(sid_to_sidx)
    train_plays.to_csv(os.path.join(data_dir, 'train.csv'), index=False)

    vad_plays = filtered_pd.loc[filtered_pd['userId'].isin(vd_users)]
    vad_plays.to_csv(os.path.join(data_dir, 'vad.csv'), index=False)

    test_plays = filtered_pd.loc[filtered_pd['userId'].isin(te_users)]
    test_plays.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
