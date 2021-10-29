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
from sklearn.utils import shuffle


class Data(object):
    def __init__(self, path):
        self.path = path
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

        self.train_set, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    for idx, i in enumerate(train_items):
                        self.R[uid, i] = 1.
                    self.train_set[uid] = train_items
                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('\nusing dataset : ', self.path)
        print('n_users= %d, n_items= %d' % (self.n_users, self.n_items))
        print('n_interactions= %d' % (self.n_train + self.n_test))
        print('n_train= %d, n_test= %d, sparsity= %.5f'
              % (self.n_train, self.n_test,
                 (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def get_user_item_adj(self):
        print('user_item bipartite shape : ', self.R.shape)
        return self.R.tocsr()

    def save_user_item_adj(self):
         # self.R is collected only from the 'train_file' !
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


    def make_train_all_mat(self):
        pos_indptr = [0]  # indptr
        pos_indices = []
        t1 = time.time()
        for user_idx in range(len(self.train_set)):
            item_seq = self.train_set[user_idx]
            pos_indptr.append(pos_indptr[-1] + len(item_seq))
            pos_indices.extend(item_seq)
        pos_data = np.ones(len(pos_indices))
        pos_mat = sp.csr_matrix((pos_data, pos_indices, pos_indptr),
                                shape=(self.n_users, self.n_items))
        t2 = time.time()
        print('\ntrain all mat making takes : ', t2 - t1)
        return pos_mat

    def make_train_sample_mat(self, num_sample):
        pos_indptr = [0]  # indptr
        pos_indices = []
        neg_indptr = [0]  # indptr
        neg_indices = []
        t1 = time.time()
        for user_idx in range(len(self.train_set)):
            item_seq = self.train_set[user_idx]
            pos_indptr.append(pos_indptr[-1] + len(item_seq))
            pos_indices.extend(item_seq)

            neg_items = list(set(range(self.n_items)) - set(item_seq))
            num_neg = num_sample - len(item_seq)
            neg_sample = rd.sample(neg_items, k=num_neg)
            # SHOULDn't use random.choices()!
            # it replaces the chosen elements
            # thus can end up with same elements in the returned sequence !!
            neg_indptr.append(neg_indptr[-1] + len(neg_sample))
            neg_indices.extend(neg_sample)

        pos_data = np.ones(len(pos_indices))
        pos_mat = sp.csr_matrix((pos_data, pos_indices, pos_indptr),
                                shape=(self.n_users, self.n_items))

        neg_data = np.repeat(-1, len(neg_indices))
        neg_mat = sp.csr_matrix((neg_data, neg_indices, neg_indptr),
                                shape=(self.n_users, self.n_items))
        t2 = time.time()
        print('\ntrain batch matrix making takes : ', t2 - t1)
        print('sample size : ', num_sample)

        print('\nper user sequence stats ...')
        item_seq_len = []
        for i in range(pos_mat.shape[0]):
            pos_sum = pos_mat.getrow(i).toarray()[0].sum()
            item_seq_len.append(pos_sum)
        print('max item sequence length : ', max(item_seq_len))
        # gowalla : 811 # yelp : 1848  # amazon-book :
        print('min item sequence length : ', min(item_seq_len))
        # gowalla : 8  # yelp : 16   # amazon-book :
        print('avg. item seq length : ', np.mean(item_seq_len))
        # gowalla : 27  # yelp : 39   # amazon-book :
        print('item seq length std : ', np.std(item_seq_len))
        # gowalla : 36  # yelp : 45   # amazon-book :
        return pos_mat, neg_mat


    def make_test_sample_mat(self, num_sample):
        pos_indptr = [0]
        pos_indices = []
        neg_indptr = [0]
        neg_indices = []
        t1 = time.time()
        for user_idx in range(len(self.test_set)):
            item_seq = self.test_set[user_idx]
            pos_indptr.append(pos_indptr[-1] + len(item_seq))
            pos_indices.extend(item_seq)

            neg_items = list(set(range(self.n_items)) - set(item_seq))
            num_neg = num_sample - len(item_seq)
            neg_sample = rd.sample(neg_items, k=num_neg)
            neg_indptr.append(neg_indptr[-1] + len(neg_sample))
            neg_indices.extend(neg_sample)

        pos_data = np.ones(len(pos_indices))
        pos_mat = sp.csr_matrix((pos_data, pos_indices, pos_indptr),
                                shape=(self.n_users, self.n_items))
        
        neg_data = np.repeat(-1, len(neg_indices))
        neg_mat = sp.csr_matrix((neg_data, neg_indices, neg_indptr),
                                shape=(self.n_users, self.n_items))
        t2 = time.time()
        print('\ntest batch matrix making takes : ', t2 - t1)
        print('sample size : ', num_sample)
        return pos_mat, neg_mat


    def make_test_all_mat(self):
        pos_indptr = [0]
        pos_indices = []
        t1 = time.time()
        for user_idx in range(len(self.test_set)):
            item_seq = self.test_set[user_idx]
            pos_indptr.append(pos_indptr[-1] + len(item_seq))
            pos_indices.extend(item_seq)
        pos_data = np.ones(len(pos_indices))
        pos_mat = sp.csr_matrix((pos_data, pos_indices, pos_indptr),
                                shape=(self.n_users, self.n_items))
        t2 = time.time()
        print('\ntest all mat making takes : ', t2 - t1)
        return pos_mat

    def load_all_mat(self):
        train_mat = self.make_train_all_mat()
        test_mat = self.make_test_all_mat()
        return train_mat, test_mat


    def make_test_sample(self, num_sample):
        train_value_list = list(self.train_set.values())
        train_item_list = [item for i in range(len(train_value_list)) for item in train_value_list[i]]
        train_item_set = set(train_item_list)
        train_items = list(train_item_set)

        test_value_list = list(self.test_set.values())
        test_item_list = [item for i in range(len(test_value_list)) for item in test_value_list[i]]
        test_item_set = set(test_item_list)
        test_items = list(test_item_set)



if __name__ == '__main__':
    # path = './data/gowalla'
    path = './data/yelp2018'
    # path = './data/amazon-book'

    data = Data(path)
    user_mat, item_mat = data.save_user_item_adj()
    # train_df = data.make_train_df()
    # test_df = data.make_test_df()
    # num_sample = 3000
    # data.save_train_df(num_sample)
    # data.save_test_df(num_sample)
