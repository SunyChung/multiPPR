import os
import pickle
import time
import numpy as np
import torch
import scipy.sparse as sp
from sknetwork.ranking import PageRank
from collections import defaultdict


class get_ppr(object):
    def __init__(self, top_k, dict):
        self.array_idxs = np.array(list(dict.values()))[:, :, :top_k]

    def reshaped_tensor(self):
        reshaped = self.array_idxs.reshape(self.array_idxs.shape[0], -1)
        return torch.LongTensor(reshaped)

class multi_PPR(object):
    def __init__(self, mat, damping_factors):
        self.mat = mat
        self.base_zeros = np.zeros(mat.shape[0] - 1)
        self.damping_factors = damping_factors
        self.per_user_idx_dict = defaultdict(dict)

    def calculate(self):
        start = time.time()
        for i in range(len(self.damping_factors)):
            pagerank = PageRank(damping_factor=damping_factors[i], n_iter=5)
            for j in range(self.mat.shape[0]):
                seed = np.insert(self.base_zeros, j, 1)
                ppr = pagerank.fit_transform(self.mat.toarray(), seeds=seed)
                idx = np.argsort(ppr)[::-1]
                per_user_idx_dict[j][damping_factors[i]] = idx
                if j % 1000 == 0:
                    print('%d nodes processed!' % j)
                    print('upto now %f S passed' % (time.time() - start))
        end = time.time()
        print('multi-ppr processing takes : ', end - start)


if __name__ == '__main__':
    data_dir = './data/gowalla'
    damping_factors = [0.30, 0.50, 0.75, 0.85, 0.95]

    # user feature extraction : 29858
    user_mat = sp.load_npz(data_dir + '/user_mat.npz')
    print('user_mat shape : ', user_mat.shape)
    per_user_idx_dict = defaultdict(dict)
    base_zeros = np.zeros(user_mat.shape[0] - 1)
    start = time.time()
    for i in range(len(damping_factors)):
        pagerank = PageRank(damping_factor=damping_factors[i], n_iter=5)
        for j in range(user_mat.shape[0]):
            seed = np.insert(base_zeros, j, 1)
            ppr = pagerank.fit_transform(user_mat.toarray(), seeds=seed)
            idx = np.argsort(ppr)[::-1]
            per_user_idx_dict[j][damping_factors[i]] = idx
            if j % 1000 == 0:
                print('%d nodes processed!' % j)
                print('upto now %f S passed' % (time.time() - start))
    end = time.time()
    print('multi-ppr processing takes : ', end - start)
    with open(os.path.join(data_dir, '/per_user_idx.dict'), 'wb') as f:
        pickle.dump(per_user_idx_dict, f)

    # item feature extraction : 40981
    # item_mat = sp.load_npz(data_dir + '/item_mat.npz')
    # print('movie_mat shape : ', item_mat.shape)
    # multi_ppr = MultiPPR(damping_factors, item_mat)
    # start = time.time()
    # per_item_idx_dict = {}
    # for i in range(item_mat.shape[0]):
    #     indices = multi_ppr.multi_context(i)
    #     per_item_idx_dict[i] = indices
    #     if i % 1000 == 0:
    #         print('%d nodes processed!' % i)
    #         print('upto now %f S passed' % (time.time() - start))
    # end = time.time()
    # print('multi-ppr processing takes : ', end - start)
    # with open(os.path.join(data_dir, '/per_item_idx.dict'), 'wb') as f:
    #     pickle.dump(per_item_idx_dict, f)
