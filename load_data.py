'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
import time
import os
import pandas as pd


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        self.R_User = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)
        # building csr_matrix from loading data takes forever ...
        # self.R = sp.csr_matrix((self.n_users, self.n_items), dtype=np.float32)
        # self.R_Item = sp.csr_matrix((self.n_items, self.n_items), dtype=np.float32)
        # self.R_User = sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    for idx, i in enumerate(train_items):
                        self.R[uid, i] = 1.
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

    def make_train_df(self):
        train_dict = self.train_items
        user_item = []
        dic_list = list(train_dict.items())

        item_values = list(train_dict.values())
        # 아래 for loop 순서 주의할 것 !
        items = [j for i in range(len(item_values)) for j in item_values[i]]
        # 근데 비교해 보니 이거 안 해도 그냥 전체 item number 그대로 임 ;
        unique_items = list(set(items))

        t1 = time.time()
        negatives = []
        for i in range(len(dic_list)):
            user_idx = dic_list[i][0]
            item_seq = dic_list[i][1]
            neg_items = list(set(unique_items) - set(item_seq))
            neg_sample = [rd.choice(neg_items) for _ in range(1000)]
            neg = np.array([np.repeat(user_idx, len(neg_sample)), neg_sample]).T
            # print('negatives shape : ', neg.shape)  # (1000, 2)
            negatives.append(neg)
            user_item.extend(list(zip(np.repeat(user_idx, len(item_seq)), item_seq)))
        t2 = time.time()
        print('takes : ', t2 - t1)
        train_df = pd.DataFrame(user_item, columns=['userId', 'itemId'])
        train_df['target'] = 1

        train_neg_df = pd.DataFrame(np.concatenate(negatives, axis=0),
                                    columns=['userId', 'itemId'])
        train_neg_df['target'] = 0

        merged = pd.concat([train_df, train_neg_df])
        shuffled_df = merged.groupby('userId').sample(frac=1)
        return shuffled_df

    def make_test_df(self):
        test_dict = self.test_set
        user_item = []
        dic_list = list(test_dict.items())
        for i in range(len(dic_list)):
            user_idx = dic_list[i][0]
            item_seq = dic_list[i][1]
            user_item.extend(list(zip(np.repeat(user_idx, len(item_seq)), item_seq)))
        test_df = pd.DataFrame(user_item, columns=['userId', 'itemId'])
        test_df['target'] = 1
        return test_df

    def save_train_df(self):
        train_df = self.make_train_df()
        train_df.to_csv(os.path.join(self.path, 'train.csv'), index=False)
        print('train data saved!')

    def save_test_df(self):
        test_df = self.make_test_df()
        test_df.to_csv(os.path.join(self.path, 'test.csv'), index=False)
        print('test data saved!')

    def load_all(self):
        train_df = pd.read_csv(os.path.join(self.path, 'train.csv'))
        train_np = train_df.to_numpy()
        test_df = pd.read_csv(os.path.join(self.path, 'test.csv'))
        test_np = test_df.to_numpy()
        return train_np, test_np

    def get_user_item_adj(self):
        print('user_item bipartite shape : ', self.R.shape)
        return self.R.tocsr()

    def project_user_item_adj(self):
        bipartite = self.R.tocsr()
        start = time.time()
        user_mat = bipartite * bipartite.transpose()
        print('projected user mat : ', user_mat.shape)
        item_mat = bipartite.transpose() * bipartite
        print('projected item mat : ', item_mat.shape)
        sp.save_npz(self.path + '/user_mat.npz', user_mat)
        sp.save_npz(self.path + '/item_mat.npz', item_mat)
        print('projection & saving takes : ', time.time() - start)
        return user_mat, item_mat


    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users= %d, n_items= %d' % (self.n_users, self.n_items))
        print('n_interactions= %d' % (self.n_train + self.n_test))
        print('n_train= %d, n_test= %d, sparsity= %.5f'
              % (self.n_train, self.n_test,
                 (self.n_train + self.n_test)/(self.n_users * self.n_items)))


if __name__ == '__main__':
    path = './data/gowalla'
    batch_size = 500

    data = Data(path, batch_size)
    # user_mat, item_mat = data.project_user_item_adj()
    train_df = data.make_train_df()
    data.save_train_df()
    data.save_test_df()
