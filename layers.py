import numpy as np
from sknetwork.ranking import PageRank
from collections import defaultdict
import pickle
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_movie_matrix


def compute_factor_distribution(data_dir, top_k):
    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'rb') as f:
        per_item_ppr_dict = pickle.load(f)

    ppr_scores = np.array(list(per_item_ppr_dict.values()))
    first_com, sec_com, third_com, four_com, fifth_com = [], [], [], [], []
    for i in range(len(ppr_scores)):
        first_com.append(np.sum(ppr_scores[i][0][0, :][:top_k]))
        sec_com.append(np.sum(ppr_scores[i][0][1, :][:top_k]))
        third_com.append(np.sum(ppr_scores[i][0][2, :][:top_k]))
        four_com.append(np.sum(ppr_scores[i][0][3, :][:top_k]))
        fifth_com.append(np.sum(ppr_scores[i][0][4, :][:top_k]))
    first_mean = np.mean(first_com)
    sec_mean = np.mean(sec_com)
    third_mean = np.mean(third_com)
    four_mean = np.mean(four_com)
    fifth_mean = np.mean(fifth_com)

    per_item_diff_dict = defaultdict(list)
    for i in range(len(ppr_scores)):
        per_item_diff_dict[i].append(np.array((first_com[i] - first_mean, sec_com[i] - sec_mean,
                                              third_com[i] - third_mean, four_com[i] - four_mean,
                                              fifth_com[i] - fifth_mean)))

    com_diff = np.array(list(per_item_diff_dict.values())) # np.shape : (3952, 1, 5)
    com_1_diff = com_diff[:, 0, 0]
    com_2_diff = com_diff[:, 0, 1]
    com_3_diff = com_diff[:, 0, 2]
    com_4_diff = com_diff[:, 0, 3]
    com_5_diff = com_diff[:, 0, 4]
    return per_item_diff_dict


class MultiPPR(object):
    def __init__(self, damping_factors, movie_mat):
        super(MultiPPR, self).__init__()
        self.damping_factors = damping_factors
        self.movie_matrix = movie_mat

    def multi_contexts(self, target_idx):
        base_zeros = np.zeros(self.movie_matrix.shape[0] - 1)
        seed = np.insert(base_zeros, target_idx, 1)
        multi_score = []
        indices = []
        for i in range(len(self.damping_factors)):
            pagerank = PageRank(damping_factor=self.damping_factors[i])
            ppr = pagerank.fit_transform(self.movie_matrix.toarray(), seeds=seed)
            idx = np.argsort(ppr)[::-1][1:]
            sorted_scores = ppr[idx]
            multi_score.append(np.array(sorted_scores))
            indices.append(np.array(idx))
        return np.array(multi_score), np.array(indices)


class ContextFeatures(object):
    def __init__(self, data_dir, top_k):
        self.data_dir = data_dir
        self.top_k = top_k

        with open(self.data_dir + 'per_item_idx.dict', 'rb') as f:
            idx_dict = pickle.load(f)
        self.idx_values = np.array(list(idx_dict.values()))  # np.shape(item_idx) : (3952, 1, 5, 3951)

        with open(self.data_dir + 'per_item_ppr.dict', 'rb') as f:
            ppr_dict = pickle.load(f)
        self.ppr_scores = np.array(list(ppr_dict.values()))

    def neighbor_embeds(self, target_idx):
        first_neighbors = torch.Tensor(self.idx_values[target_idx, :, 0, :self.top_k])
        sec_neighbors = torch.Tensor(self.idx_values[target_idx, :, 1, :self.top_k])
        third_neighbors = torch.Tensor(self.idx_values[target_idx, :, 2, :self.top_k])
        four_neighbors = torch.Tensor(self.idx_values[target_idx, :, 3, :self.top_k])
        fif_neighbors = torch.Tensor(self.idx_values[target_idx, :, 4, :self.top_k])

        first_scores = torch.Tensor(self.ppr_scores[target_idx, :, 0, :self.top_k])
        sec_scores = torch.Tensor(self.ppr_scores[target_idx, :, 1, :self.top_k])
        third_scores = torch.Tensor(self.ppr_scores[target_idx, :, 2, :self.top_k])
        four_scores = torch.Tensor(self.ppr_scores[target_idx, :, 3, :self.top_k])
        fif_scores = torch.Tensor(self.ppr_scores[target_idx, :, 4, :self.top_k])
        return [first_neighbors, first_scores, sec_neighbors, sec_scores, third_neighbors, third_scores,
                four_neighbors, four_scores, fif_neighbors, fif_scores]


class ItemLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ItemLinear, self).__init__()
        self.ln1 = nn.Linear(input_dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, input_dim)
        self.final = nn.Linear(input_dim, output_dim)

    def forward(self, index, score):
        x = F.relu(self.ln1(index))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        x = torch.mul(x, score)
        x = self.final(x)
        return x.flatten()


if __name__ == '__main__':
    data_dir = './data/ml-1m/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    damping_factors = [0.30, 0.50, 0.70, 0.85, 0.95]
    movie_mat = get_movie_matrix(data_dir)
    multi_ppr = MultiPPR(damping_factors, movie_mat)

    start = time.time()
    # the defaul dictionary type is sufficient 'cause the return values already has multi array values !!!
    per_item_ppr_dict = defaultdict(list)
    # per_item_ppr_dict = {}
    per_item_idx_dict = defaultdict(list)
    # per_item_idx = dict = {}
    for i in range(movie_mat.shape[0]):
        scores, indices = multi_ppr.multi_contexts(i)
        per_item_ppr_dict[i].append(scores)
        # per_item_per_dict[i] = scores
        per_item_idx_dict[i].append(indices)
        # per_item_idx_dict[i[ = indices
        if i % 10 == 0:
            print('%d nodes processed!' % i)
    end = time.time()
    print('multi-ppr processing takes : ', end - start)

    with open(os.path.join(data_dir, 'per_item_ppr.dict'), 'wb') as f:
        pickle.dump(per_item_ppr_dict, f)
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'wb') as f:
        pickle.dump(per_item_idx_dict, f)
