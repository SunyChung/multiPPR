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
        # depending on the pandas version the below ['size'] returns error message !
        # this works for pandas = 1.1.2 or later versions
        tp = tp[tp['movieId'].isin(item_count.index[item_count['size'] >= min_sc])]
        print('movieId filtered !')

    if min_uc > 0:
        user_count = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(user_count.index[user_count['size'] >= min_uc])]
        print('userId filtered!')

    print('filtered "tp" shape : ', tp.shape)
    user_count, item_count = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, user_count, item_count


def data_filtering(data_dir, raw_data):
    filtered_pd, user_activity, item_popularity = filter_triplets(raw_data)
    sparsity = 1. * filtered_pd.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    print('after filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)'
          % (filtered_pd.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))
    # after filtering, there are 574548 watching events from 6031 users and 3533 movies (sparsity: 2.696%)

    filtered_pd = filtered_pd[['userId', 'movieId']]
    with open(data_dir + 'filtered_ratings.csv', 'w') as f:
        filtered_pd.to_csv(f, index=False)
    return filtered_pd, user_activity, item_popularity


def split_train_test_proportion(data, test_prop):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = [], []
    np.random.seed(1324)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)
        if i % 100 == 0:
            print('%d users sampled' %i)
            sys.stdout.flush()
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    return data_tr, data_te


def numbered(tp, movie2id, user2id):
    uid = tp['userId'].apply(lambda x: user2id[x])
    sid = tp['movieId'].apply(lambda x: movie2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


if __name__ == '__main__':
    data_dir = './data/ml-1m/'
    threshold = 3.5
    n_held_out_users = 500  # (50, 500, 10000, 40000)
    test_prop = 0.2

    raw_data = pd.read_csv(data_dir + 'ratings.dat', sep='::',
                           names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
    raw_data = raw_data[raw_data['rating'] > threshold]  # [575281 rows x 4 columns]
    print('raw_data rating filtered : ', raw_data.shape)

    filtered_pd, user_count, item_count = data_filtering(data_dir, raw_data)

    ######################### userId - idx mapping #############################
    # uidx 는 0 부터 순차적으로 증가하는 index 값 (user_count, 즉, pandas groupby 로 return 되는 값)
    # unique_uidx : filtering 된 전체 userId index
    # -> train, validation, test set splitting 에 사용해야 함
    unique_uidx = user_count.index
    print('number of total users : ', len(unique_uidx))
    with open(os.path.join(data_dir, 'unique_uidx'), 'wb') as f:
        pickle.dump(unique_uidx, f)

    # uidx 와 uid 간의 mapping dictionary !
    uidx_to_uid = user_count['userId'].to_dict()
    with open(data_dir + 'uidx_to_uid.dict', 'wb') as f:
        pickle.dump(uidx_to_uid, f)
    uid_to_uidx = {value: key for (key, value) in uidx_to_uid.items()}
    with open(data_dir + 'uid_to_uidx.dict', 'wb') as f:
        pickle.dump(uid_to_uidx, f)
    ######################### userId - idx mapping #############################

    np.random.seed(1324)
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

    ######################### movieId - idx mapping ############################
    # 어차피 train, validation, test set 은 모두 train 에 들어 있는 item ID 만 사용함 !!
    # 그러니, 전체 item ID mapping 할 필요없음 !
    # train_sid 만 item 으로 남게 됨
    ######################### movieId - idx mapping ############################
    train_plays = filtered_pd.loc[filtered_pd['userId'].isin(tr_users)]
    train_sid = pd.unique(train_plays['movieId'])
    print('length of unique train items : ', len(train_sid))   # 3505
    with open(os.path.join(data_dir, 'train_sid'), 'wb') as f:
        pickle.dump(train_sid, f)

    # sid (1 부터 시작 !) 와 sidx (0 부터 시작 !)
    sid_to_sidx = {sid: i for (i, sid) in enumerate(train_sid)}
    with open(os.path.join(data_dir, 'sid_to_sidx.dict'), 'wb') as f:
        pickle.dump(sid_to_sidx, f)
    sidx_to_sid = {value: key for (key, value) in sid_to_sidx.items()}
    with open(os.path.join(data_dir, 'sidx_to_sid.dict'), 'wb') as f:
        pickle.dump(sidx_to_sid, f)

    train_data = numbered(train_plays, sid_to_sidx, uid_to_uidx)
    train_data.to_csv(os.path.join(data_dir, 'train.csv'), index=False)

    vad_plays = filtered_pd.loc[filtered_pd['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(train_sid)]
    vad_data = numbered(vad_plays, sid_to_sidx, uid_to_uidx)
    vad_data.to_csv(os.path.join(data_dir, 'vad.csv'), index=False)

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays, test_prop)
    vad_data_tr = numbered(vad_plays_tr, sid_to_sidx, uid_to_uidx)
    vad_data_te = numbered(vad_plays_te, sid_to_sidx, uid_to_uidx)
    # print(vad_data_tr)
    # print(vad_data_te)
    vad_data_tr.to_csv(os.path.join(data_dir, 'vad_tr.csv'), index=False)
    vad_data_te.to_csv(os.path.join(data_dir, 'vad_te.csv'), index=False)

    test_plays = filtered_pd.loc[filtered_pd['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(train_sid)]
    test_data = numbered(test_plays, sid_to_sidx, uid_to_uidx)
    test_data.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays, test_prop)
    test_data_tr = numbered(test_plays_tr, sid_to_sidx, uid_to_uidx)
    test_data_te = numbered(test_plays_te, sid_to_sidx, uid_to_uidx)
    # print(test_data_tr)
    # print(test_data_te)
    test_data_tr.to_csv(os.path.join(data_dir, 'test_tr.csv'), index=False)
    test_data_te.to_csv(os.path.join(data_dir, 'test_te.csv'), index=False)
