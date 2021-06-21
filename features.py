import os
import time
import pickle
import numpy as np
from sknetwork.ranking import PageRank
import torch
import torch.nn as nn

from utils import *


class PPRfeatures(object):
    def __init__(self, data_dir, top_k, idx_dict, ppr_dict):
        self.data_dir = data_dir
        self.top_k = top_k
        self.idx_values = np.array(list(idx_dict.values()))
        self.ppr_scores = np.array(list(ppr_dict.values()))

    def idx_tensor(self):
        # dictionary shape : [all nodes idxs , multi-factors, ranked idxs]
        idx_top_k = self.idx_values[:, :, :self.top_k]
        idx_tensor = torch.LongTensor(idx_top_k.reshape(idx_top_k.shape[0], -1))
        # [all nodes idxs, multi-factors x top-k = 5 x 20]
        # print('idx_tensor shape : ', idx_tensor.shape)  # torch.Size([3505, 100])
        return idx_tensor

    def score_tensor(self):
        # dictionary shape : [all nodes idxs, multi-factors, ranked scores]
        score_top_k = self.ppr_scores[:, :, :self.top_k]
        score_tensor = torch.FloatTensor(score_top_k.reshape(score_top_k.shape[0], -1))
        # [all nodes idxs, multi-factors x top-k = 5 x 20]
        # print('score_tensor shape : ', score_tensor.shape)  # torch.Size([3505, 100])
        return score_tensor

    # previously tried with nn.Embedding
    # idx_embeds got torch.FloatTensor(Tensor) as the dtype,
    # thus, can not be used as the another input to the nn.Embedding
    def idx_embeds(self):
        idx_top_k = self.idx_values[:, :, 1:self.top_k+1]
        idx_emb = nn.Embedding.from_pretrained(
            torch.Tensor(idx_top_k.reshape(idx_top_k.shape[0], -1)), freeze=True)
        return idx_emb

    def score_embeds(self):
        score_top_k = self.ppr_scores[:, :, 1:self.top_k+1]
        score_emb = nn.Embedding.from_pretrained(
            torch.Tensor(score_top_k.reshape(score_top_k.shape[0], -1)), freeze=True)
        return score_emb


class OneFactorFeature(object):
    def __init__(self, data_dir, top_k, idx_dict, ppr_dict):
        self.data_dir = data_dir
        self.top_k = top_k
        self.idx_values = np.array(list(idx_dict.values()))
        self.ppr_scores = np.array(list(ppr_dict.values()))

    def idx_tensor(self):
        idx_top_k = self.idx_values[:, :self.top_k]
        # print('idx_top_k shape : ', idx_top_k.shape)  # (3503, 50), (6031, 50)
        idx_tensor = torch.LongTensor(idx_top_k.reshape(idx_top_k.shape[0], -1))
        # print('idx_tensor shape : ', idx_tensor.shape)  # torch.Size([3503, 50]), torch.Size([6031, 50])
        # [all nodes idxs, top-k]
        return idx_tensor

    def ppr_tensor(self):
        ppr_top_k = self.ppr_scores[:, :self.top_k]
        ppr_tensor = torch.FloatTensor(ppr_top_k.reshape(ppr_top_k.shape[0], -1))
        # should be [all nodes idxs, top-k]
        return ppr_tensor

'''
if __name__ == '__main__':
    # data_dir = './data/ml-1m'
    data_dir = './data/ml-20m/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    top_k = 20  # 10
    multi_factor = 5

    # item context tensor
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'rb') as f:
        item_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
        item_ppr_dict = pickle.load(f)

    pf = PPRfeatures(data_dir, top_k, item_idx_dict, item_ppr_dict)
    item_idx_tensor = pf.idx_tensor()
    item_scr_tensor = pf.score_tensor()
    del item_idx_dict
    del item_ppr_dict

    # user context tensor
    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'rb') as f:
        user_idx_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'rb') as f:
        user_ppr_dict = pickle.load(f)

    pf = PPRfeatures(data_dir, top_k, user_idx_dict, user_ppr_dict)
    user_idx_tensor = pf.idx_tensor()
    user_scr_tensor = pf.score_tensor()
    del user_idx_dict
    del user_ppr_dict
'''


class MultiPPR(object):
    def __init__(self, damping_factors, matrix):
        super(MultiPPR, self).__init__()
        self.damping_factors = damping_factors
        self.mat = matrix

    def multi_contexts(self, target_idx):
        base_zeros = np.zeros(self.mat.shape[0] - 1)
        seed = np.insert(base_zeros, target_idx, 1)
        multi_score = []
        indices = []
        for i in range(len(self.damping_factors)):
            pagerank = PageRank(damping_factor=self.damping_factors[i])
            ppr = pagerank.fit_transform(self.mat.toarray(), seeds=seed)
            # don't need to exclude the first element from the start
            # can filtering the idx afterwards, if necessary
            idx = np.argsort(ppr)[::-1]
            sorted_scores = ppr[idx]
            multi_score.append(np.array(sorted_scores))
            indices.append(np.array(idx))
        return np.array(multi_score), np.array(indices)


class GetPPR(object):
    def __init__(self, damping_factor, matrix):
        super(GetPPR, self).__init__()
        self.damping_factor = damping_factor
        self.mat = matrix

    def get_score_and_index(self, target_idx):
        base_zeros = np.zeros(self.mat.shape[0] - 1)
        seed = np.insert(base_zeros, target_idx, 1)
        pagerank = PageRank(damping_factor=self.damping_factor)
        ppr = pagerank.fit_transform(self.mat.toarray(), seeds=seed)
        idxs = np.argsort(ppr)[::-1]
        sorted_scores = ppr[idxs]
        return sorted_scores, idxs


if __name__ == '__main__':
    # one factor only
    data_dir = './data/ml-1m'
    # data_dir = './data/ml-20m'
    damping_factor = 0.85

    # item features
    movie_mat = get_movie_matrix(data_dir)
    print('movie_mat shape : ', movie_mat.shape)
    movie_ppr = GetPPR(damping_factor, movie_mat)
    start = time.time()
    per_item_ppr_dict = {}
    per_item_idx_dict = {}
    for i in range(movie_mat.shape[0]):
        score, index = movie_ppr.get_score_and_index(i)
        per_item_ppr_dict[i] = score
        per_item_idx_dict[i] = index
        if i % 500 == 0:
            print('%d nodes processed!' %i)
            print('upto now %f seconds have passed' %(time.time() - start))
    end = time.time()
    print('PPR processing takes : ', end - start)
    with open(os.path.join(data_dir, 'item_ppr_' + str(damping_factor) + '.dict'), 'wb') as f:
        pickle.dump(per_item_ppr_dict, f)
    with open(os.path.join(data_dir, 'item_idx_' + str(damping_factor) + '.dict'), 'wb') as f:
        pickle.dump(per_item_idx_dict, f)

    # user features
    user_mat = get_user_matrix(data_dir)
    print('user_mat shape : ', user_mat.shape)
    movie_ppr = GetPPR(damping_factor, user_mat)
    start = time.time()
    per_item_ppr_dict = {}
    per_item_idx_dict = {}
    for i in range(user_mat.shape[0]):
        score, index = movie_ppr.get_score_and_index(i)
        per_item_ppr_dict[i] = score
        per_item_idx_dict[i] = index
        if i % 500 == 0:
            print('%d nodes processed!' %i)
            print('upto now %f seconds have passed' %(time.time() - start))
    end = time.time()
    print('PPR processing takes : ', end - start)
    with open(os.path.join(data_dir, 'user_ppr_' + str(damping_factor) + '.dict'), 'wb') as f:
        pickle.dump(per_item_ppr_dict, f)
    with open(os.path.join(data_dir, 'user_idx_' + str(damping_factor) + '.dict'), 'wb') as f:
        pickle.dump(per_item_idx_dict, f)

    # multi-factor feature extraction
    '''
    # for multiple damping factor calculation
    damping_factors = [0.30, 0.50, 0.70, 0.85, 0.95]
    movie_mat = get_movie_matrix(data_dir)
    print('movie_mat shape : ', movie_mat.shape)
    multi_ppr = MultiPPR(damping_factors, movie_mat)
    start = time.time()
    per_item_ppr_dict = {}
    per_item_idx_dict = {}
    for i in range(movie_mat.shape[0]):
        scores, indices = multi_ppr.multi_contexts(i)
        per_item_ppr_dict[i] = scores
        per_item_idx_dict[i] = indices
        if i % 1000 == 0:
            print('%d nodes processed!' % i)
            print('upto now %f seconds passed' % (time.time() - start))
    end = time.time()
    print('multi-ppr processing takes : ', end - start)
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'wb') as f:
        pickle.dump(per_item_ppr_dict, f)
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'wb') as f:
        pickle.dump(per_item_idx_dict, f)

    user_mat = get_user_matrix(data_dir)
    print('user_mat shape : ', user_mat.shape)
    multi_ppr = MultiPPR(damping_factors, user_mat)

    start = time.time()
    # the default dictionary type is sufficient 'cause the return values already has multi array values !!!
    per_user_ppr_dict = {}
    per_user_idx_dict = {}
    for i in range(user_mat.shape[0]):
        scores, indices = multi_ppr.multi_contexts(i)
        per_user_ppr_dict[i] = scores
        per_user_idx_dict[i] = indices
        if i % 300 == 0:
            print('%d nodes processed!' % i)
            print('upto now %f seconds passed' % (time.time() - start))
    end = time.time()
    print('multi-ppr processing takes : ', end - start)

    with open(os.path.join(data_dir, 'per_user_ppr.dict'), 'wb') as f:
        pickle.dump(per_user_ppr_dict, f)
    with open(os.path.join(data_dir, 'per_user_idx.dict'), 'wb') as f:
        pickle.dump(per_user_idx_dict, f)
    '''
