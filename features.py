import os
import pickle
import time
import numpy as np
import torch
from sknetwork.ranking import PageRank

from utils import get_movie_matrix, get_user_matrix


class PPRfeatures(object):
    def __init__(self, top_k, idx_dict, ppr_dict):
        self.idx_values = np.array(list(idx_dict.values()))[:, :, :top_k]
        # print('idx_values shape : ', self.idx_values.shape)  # (3515, 5, 50)
        self.ppr_scores = np.array(list(ppr_dict.values()))[:, :, :top_k]

    def reshaped_idx_tensor(self):
        reshaped = self.idx_values.reshape(self.idx_values.shape[0], -1)
        idx_tensor = torch.LongTensor(reshaped)
        # print('idx_tensor shape : ', idx_tensor.shape)  # torch.Size([3515, 250])
        # idx_tensor = torch.LongTensor(self.idx_values)
        # [# items/users, multi-factors, top_k]
        return idx_tensor

    def reshaped_scr_tensor(self):
        reshaped = self.ppr_scores.reshape(self.ppr_scores.shape[0], -1)
        scr_tensor = torch.FloatTensor(reshaped)
        # print('scr_tensor shape : ', scr_tensor.shape)  # torch.Size([3515, 250])
        # scr_tensor = torch.LongTensor(self.ppr_scores)
        # [# items/users, multi-factors, top_k]
        return scr_tensor


'''
if __name__ == '__main__':
    data_dir = './data/ml-1m'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    top_k = 20

    # item context tensor
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
        item_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
        item_ppr_dict = pickle.load(f)
    pf = PPRfeatures(top_k, item_idx_dict, item_ppr_dict)
    item_idx_tensor = pf.idx_tensor()
    item_scr_tensor = pf.scr_tensor()
    del item_idx_dict
    del item_ppr_dict

    # user context tensor
    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
        user_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
        user_ppr_dict = pickle.load(f)
    pf = PPRfeatures(top_k, user_idx_dict, user_ppr_dict)
    user_idx_tensor = pf.idx_tensor()
    user_scr_tensor = pf.scr_tensor()
    del user_idx_dict
    del user_ppr_dict
'''


class MultiPPR(object):
    def __init__(self, damping_factors, matrix):
        super(MultiPPR, self).__init__()
        self.damping_factors = damping_factors
        self.mat = matrix

    def multi_context(self, target_idx):
        base_zeros = np.zeros(self.mat.shape[0] - 1)
        seed = np.insert(base_zeros, target_idx, 1)
        multi_score = []
        indices = []
        for i in range(len(self.damping_factors)):
            pagerank = PageRank(damping_factor=self.damping_factors[i])
            ppr = pagerank.fit_transform(self.mat.toarray(), seeds=seed)
            idx = np.argsort(ppr)[::-1]
            sorted_scroes = ppr[idx]
            multi_score.append(np.array(sorted_scroes))
            indices.append(np.array(idx))
        return np.array(multi_score), np.array(indices)


if __name__ == '__main__':
    data_dir = './data/ml-1m'
    # data_dir = './data/ml-20m'
    damping_factors = [0.30, 0.50, 0.75, 0.85, 0.95]

    # item feature extraction
    movie_mat = get_movie_matrix(data_dir)
    print('movie_mat shape : ', movie_mat.shape)
    multi_ppr = MultiPPR(damping_factors, movie_mat)
    start = time.time()
    per_item_ppr_dict = {}
    per_item_idx_dict = {}
    for i in range(movie_mat.shape[0]):
        scores, indices = multi_ppr.multi_context(i)
        per_item_ppr_dict[i] = scores
        per_item_idx_dict[i] = indices
        if i % 100 == 0:  # for ml-1m
        # if i % 10 == 0:  # for ml-20m
            print('%d nodes processed!' % i)
            print('upto now %f S passed' % (time.time() - start))
    end = time.time()
    print('multi-ppr processing takes : ', end - start)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'wb') as f:
        pickle.dump(per_item_ppr_dict, f)
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'wb') as f:
        pickle.dump(per_item_idx_dict, f)

    # user feature extraction
    user_mat = get_user_matrix(data_dir)
    print('user_mat shape : ', user_mat.shape)
    multi_ppr = MultiPPR(damping_factors, user_mat)
    start = time.time()
    per_user_ppr_dict = {}
    per_user_idx_dict = {}
    for i in range(user_mat.shape[0]):
        scores, indices = multi_ppr.multi_context(i)
        per_user_ppr_dict[i] = scores
        per_user_idx_dict[i] = indices
        if i % 100 == 0:
        # if i % 10 == 0:  # for ml-20m
            print('%d nodes processed!' % i)
            print('upto now %f S passed' % (time.time() - start))
    end = time.time()
    print('multi-ppr processing takes : ', end - start)
    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'wb') as f:
        pickle.dump(per_user_ppr_dict, f)
    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'wb') as f:
        pickle.dump(per_user_idx_dict, f)
