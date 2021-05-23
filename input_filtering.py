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
    # after filtering, there are 472443 watching events from 6036 users and 2778 movies (sparsity: 2.818%)

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

    # unique_uid 는 filtered_pd 로, rating >= 3.5, min_uc = 5, min_sc = 0 으로 걸러진 userId
    # userId 는 한 번 filtering 되고 나서 train, validation, test 셋으로 나뉘기 때문에
    # filtered_pd, user_count 로 unique_uid 를 만듬
    unique_uidx = user_count.index
    with open(os.path.join(output_dir, 'unique_uidx'), 'wb') as f:
        pickle.dump(unique_uidx, f)

    uidx_to_uid = user_count['userId'].to_dict()
    with open(output_dir + 'uidx_to_uid.dict', 'wb') as f:
        pickle.dump(uidx_to_uid, f)

    uid_to_uidx = {value: key for (key, value) in uidx_to_uid.items()}
    with open(output_dir + 'uid_to_uidx.dict', 'wb') as f:
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

    # unique_sid 를 filtered_pd, tr_users 에서 뽑은 train_plays 로만 지정한 것은,
    # train 에 포함되지 않는 movieId 는 어차피 validation, test 에도 사용 안 한다는 것
    # 그러면 multiple PPR 추출할 bipartite matrix 도 train-only movieId 를 써야 하나 ?
    '''
    train_plays = filtered_pd.loc[filtered_pd['userId'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['movieId'])
    # print('sorted unique_sid : ', sorted(unique_sid))  # starts from 1 'cause it is pd.unique() array
    with open(os.path.join(output_dir, 'unique_sid'), 'wb') as f:
        pickle.dump(unique_sid, f)
    '''
    unique_sidx = item_count.index
    with open(os.path.join(output_dir, 'unique_sidx'), 'wb') as f:
    pickle.dump(unique_sidx, f)

    sidx_to_sid = item_count['movieId'].to_dict()
    with open(output_dir + 'midx_to_mid.dict', 'wb') as f:
        pickle.dump(sidx_to_sid, f)

    sid_to_sidx = {value: key for (key, value) in sidx_to_sid.items()}
    with open(output_dir + 'mid_to_midx.dict', 'wb') as f:
        pickle.dump(sid_to_sidx, f)
        
        
    # train 에 나온 item 만 테스트 하도록 train movieId 만 따로 뽑기 위해 필요한 sidx..
    # 근데, 이 id 도 dictionary mapping 을 해야 
    # global id mapping 된 값들이랑 안 섞일텐데 ??
    train_sidx = pd.unique(train_plays['movieId'])
    train_mapped = [sid_to_sidx[x] for x in train_sidx]
    with open(os.path.join(output_dir, 'train_mapped_id'), 'wb') as f:
		pickle.dump(train_mapped_id, f)

    
    '''
    # 여기서 dictionary 의 역할은 0 부터 시작하는 임의의 값으로 바꿔주는 것
    # 근데, 이미 숫자인 값들을 굳이 다시 mapping 해야 하나 ?!
    # 어차피 전체 id 다 쓸 거면, 나는 굳이 mapping 안 해도 될 듯 ......
    movie2id = dict(((sid, i) for (i, sid) in enumerate(unique_sid)))
    user2id = dict(((uid, i) for (i, uid) in enumerate(unique_uid)))
    with open(os.path.join(output_dir, 'mid_to_midx.dict'), 'wb') as f:
        pickle.dump(movie2id, f)
    midx_to_mid = {value: key for (key, value) in movie2id.items()}
    with open(os.path.join(output_dir, 'midx_to_mid.dict'), 'wb') as f:
        pickle.dump(midx_to_mid, f)
    '''
    
    train_data = numbered(train_plays, sid_to_sidx, uid_to_uidx)
    # train_data = numbered(train_plays, movie2id, user2id)
    # print(train_data)
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

    vad_plays = filtered_pd.loc[filtered_pd['userId'].isin(vd_users)]
    # train 된 item index 만 쓸 것인가 ?
    # vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sidx)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(train_mapped_id)]
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays, test_prop)
    
    vad_data_tr = numbered(vad_plays_tr, sid_to_sidx, uid_to_uidx)
    vad_data_te = numbered(vad_plays_tr, sid_to_sidx, uid_to_uidx)
    # vad_data_tr = numbered(vad_plays_tr, movie2id, user2id)
    # vad_data_te = numbered(vad_plays_tr, movie2id, user2id)
    # print(vad_data_tr)
    # print(vad_data_te)
    vad_data_tr.to_csv(os.path.join(output_dir, 'vad_tr.csv'), index=False)
    vad_data_te.to_csv(os.path.join(output_dir, 'vad_te.csv'), index=False)

    test_plays = filtered_pd.loc[filtered_pd['userId'].isin(te_users)]
    # 여기도 마찬가지 ! train 된 item index 만 쓸 것인가 ?
    # test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sidx)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(train_mapped_id)]
    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays, test_prop)

    test_data_tr = numbered(test_plays_tr, sid_to_sidx, uid_to_uidx)
    test_data_te = numbered(test_plays_te, sid_to_sidx, uid_to_uidx)
    #test_data_tr = numbered(test_plays_tr, movie2id, user2id)
    #test_data_te = numbered(test_plays_te, movie2id, user2id)
    # print(test_data_tr)
    # print(test_data_te)
    test_data_tr.to_csv(os.path.join(output_dir, 'test_tr.csv'), index=False)
    test_data_te.to_csv(os.path.join(output_dir, 'test_te.csv'), index=False)

