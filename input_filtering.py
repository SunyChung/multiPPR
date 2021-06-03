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
        tp = tp[tp['movieId'].isin(item_count.index[item_count['size'] >= min_sc])]
        print('movieId filtered !')

    if min_uc > 0:
        user_count = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(user_count.index[user_count['size'] >= min_uc])]
        print('userId filtered!')

    print('filtered "tp" shape : ', tp.shape)
    user_count, item_count = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, user_count, item_count


def data_filtering(output_dir, raw_data):
    filtered_pd, user_activity, item_popularity = filter_triplets(raw_data)
    sparsity = 1. * filtered_pd.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    print('after filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)'
          % (filtered_pd.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))
    # after filtering, there are 574548 watching events from 6031 users and 3533 movies (sparsity: 2.696%)

    filtered_pd = filtered_pd[['userId', 'movieId']]
    with open(output_dir + 'filtered_ratings.csv', 'w') as f:
        filtered_pd.to_csv(f, index=False)
    return filtered_pd, user_activity, item_popularity


def split_train_test_proportion(data, test_prop):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = [], []
    np.random.seed(1234)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        '''
        if i % 100 == 0:
            print('%d users sampled' %i)
            sys.stdout.flush()
        '''
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    return data_tr, data_te


def numbered(tp, movie2id, user2id):
    uid = tp['userId'].apply(lambda x: user2id[x])
    sid = tp['movieId'].apply(lambda x: movie2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


if __name__ == '__main__':
    output_dir = './data/ml-1m/'
    threshold = 3.5
    n_held_out_users = 500  # (50, 500, 10000, 40000)
    test_prop = 0.2

    raw_data = pd.read_csv(output_dir + 'ratings.dat', sep='::',
                           names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
    raw_data = raw_data[raw_data['rating'] > threshold]  # [575281 rows x 4 columns]
    print('raw_data rating filtered : ', raw_data.shape)

    filtered_pd, user_count, item_count = data_filtering(output_dir, raw_data)

    ######################### userId - idx mapping #############################
    # unique_uidx 는 filtered_pd 로, rating >= 3.5, min_uc = 5, min_sc = 0 으로 걸러진 userId
    # userId 는 한 번 filtering 되고 나서 train, validation, test 셋으로 나뉘기 때문에
    # filtered_pd, user_count 로 unique_uidx 를 만듬
    # CAUTION !
    # user_count, 즉, groupby 로 return 되는 값을 uidx 로 사용하기 때문에 uidx 는 0 부터 시작 됨 !!
    # uidx 는 0 부터 순차적으로 증가하는 index 값
    # 실제 userId 는 uid 로 지칭함 !!
    # unique_uidx : filtering 된 전체 userId
    # -> train, validation, test set splitting 에 사용하기 위한 idx
    unique_uidx = user_count.index
    print('number of total users : ', len(unique_uidx))
    with open(os.path.join(output_dir, 'unique_uidx'), 'wb') as f:
        pickle.dump(unique_uidx, f)
    # uidx 와 uid 간의 mapping dictionary !
    uidx_to_uid = user_count['userId'].to_dict()
    with open(output_dir + 'uidx_to_uid.dict', 'wb') as f:
        pickle.dump(uidx_to_uid, f)

    uid_to_uidx = {value: key for (key, value) in uidx_to_uid.items()}
    with open(output_dir + 'uid_to_uidx.dict', 'wb') as f:
        pickle.dump(uid_to_uidx, f)
    ######################### userId - idx mapping #############################

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

    ######################### movieId - idx mapping ############################
    # unique_sid 를 filtered_pd, tr_users 에서 뽑은 train_plays 로만 지정한 것은,
    # train 에 포함되지 않는 movieId 는 어차피 validation, test 에도 사용 안 한다는 것
    # 그러면 multiple PPR 추출할 bipartite matrix 도 train-only movieId 를 써야 하나 ?  YES !!
    #
    # 		# train 에 포함된 movieId 만 가지고 bipartite -> item matrix 만들어야
    # 		# 일종의 cheating 이 안 될 것이라 생각
    # train set 에 포함되지 않은 전체 item index 값도 가지고 있을 것
    # validation, test set 에는 train data 에 있는 item 만 넣는다고 하더라도
    # 나중에 전체 graph ID 찾으려면 전체 item dictionary 가 필요하려나 싶어서...

    # 근데, 어차피 train, validation, test set 은 모두 train 에 들어 있는 item ID 만 사용함 !!
    unique_sidx = item_count.index
    print('number of total items : ', len(unique_sidx))
    with open(os.path.join(output_dir, 'unique_sidx'), 'wb') as f:
        pickle.dump(unique_sidx, f)

    sidx_to_sid = item_count['movieId'].to_dict()
    with open(output_dir + 'midx_to_mid.dict', 'wb') as f:
        pickle.dump(sidx_to_sid, f)

    sid_to_sidx = {value: key for (key, value) in sidx_to_sid.items()}
    with open(output_dir + 'mid_to_midx.dict', 'wb') as f:
        pickle.dump(sid_to_sidx, f)
    ######################### movieId - idx mapping ############################

    train_plays = filtered_pd.loc[filtered_pd['userId'].isin(tr_users)]
    # train_data = numbered(train_plays, movie2id, user2id)
    train_data = numbered(train_plays, sid_to_sidx, uid_to_uidx)
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

    # train_sidx 에 해당하는 item ID 만 가지도록,
    # validation, test set 은 (1) 먼저 validation, test user ID 로 filtering 하고
    # (2) train_sidx 에 속하는 item ID 값으로 다시 한 번 더 ! filtering 해야 함 !!
    # 근데, 다시 보니, train_sidx 는 잘 구했는데, 이걸 다시 idx 로 mapping 하고 나서
    # validation, test 에서 또 다시 numbered 시켰음 !!! 중복 !! -_

    # 일단 데이터를 filtering 하고, 저장할 때, 최종적으로 numbered 로 dictionary mapping 해야 함 !!!
    train_sidx = pd.unique(train_plays['movieId'])
    train_mapped_id = [sid_to_sidx[x] for x in train_sidx]
    with open(os.path.join(output_dir, 'train_mapped_id'), 'wb') as f:
        pickle.dump(train_mapped_id, f)

    vad_plays = filtered_pd.loc[filtered_pd['userId'].isin(vd_users)]
    # 원래 vae_cf 등에서 데이터 splitting 할 때도, 전체 item ID (=unique_sid) 안 썼음 !!
    # vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
    # 아래처럼 하면, 나중에 numbered 까지 2번 mapping 됨 -_
    # vad_plays = vad_plays.loc[vad_plays['movieId'].isin(train_mapped_id)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(train_sidx)]
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays, test_prop)
    vad_data_tr = numbered(vad_plays_tr, sid_to_sidx, uid_to_uidx)
    vad_data_te = numbered(vad_plays_tr, sid_to_sidx, uid_to_uidx)
    # print(vad_data_tr)
    # print(vad_data_te)
    vad_data_tr.to_csv(os.path.join(output_dir, 'vad_tr.csv'), index=False)
    vad_data_te.to_csv(os.path.join(output_dir, 'vad_te.csv'), index=False)

    test_plays = filtered_pd.loc[filtered_pd['userId'].isin(te_users)]
    # 원래 vae_cf 등에서 데이터 splitting 할 때도, 전체 item ID (=unique_sid) 안 썼음 !!
    # test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
    # 아래처럼 하면, 나중에 numbered 까지 2번 mapping 됨 -_
    # test_plays = test_plays.loc[test_plays['movieId'].isin(train_mapped_id)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(train_sidx)]
    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays, test_prop)
    test_data_tr = numbered(test_plays_tr, sid_to_sidx, uid_to_uidx)
    test_data_te = numbered(test_plays_te, sid_to_sidx, uid_to_uidx)
    # print(test_data_tr)
    # print(test_data_te)
    test_data_tr.to_csv(os.path.join(output_dir, 'test_tr.csv'), index=False)
    test_data_te.to_csv(os.path.join(output_dir, 'test_te.csv'), index=False)
