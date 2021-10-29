"""PageRank analysis of graph structure. """
# https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank_scipy
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import time
import os
import pickle
import pandas as pd
import sys
import gc


def pagerank_scipy(M, N, target, alpha=0.85, max_iter=30, tol=1.0e-6):
    # initial ranking vector
    x = np.repeat(1.0 / N, N)
    # personalization
    base_zeros = np.zeros(N - 1)
    p = np.insert(base_zeros, target, 1)

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        # tolerence 낮추는 게 더 빠름 !! 1.0e-6 -> 1.0e-4
        # tolerance 가 낮아지면 PageRank 값이 모두 동일해짐, 주의 !
        x = alpha * (x * M) + (1 - alpha) * p
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            return np.array(np.argsort(x)[::-1][:200])
            # return np.array(np.argsort(x)[::-1][:500])
            # return np.array(np.argsort(x)[::-1][:1000])
            # return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)


if __name__ == '__main__':
    # data_dir = './data/gowalla'
    # data_dir = './data/yelp2018'
    data_dir = './data/amazon-book'
    damping_factors = [0.30, 0.50, 0.75, 0.85, 0.95]

    # # user PagaRank : gowalla = 29,858, yelp = 31,668, amazon = 52,643
    # user_mat = sp.sparse.load_npz(data_dir + '/user_mat.npz')
    # user_pd = pd.read_csv(data_dir + 'user_list.txt', sep=' ')
    # nodelist = user_pd['remap_id'].to_list()
    # M = user_mat
    # S = np.array(M.sum(axis=1)).flatten()
    # S[S != 0] = 1.0 / S[S != 0]
    # Q = sp.sparse.spdiags(S.T, 0, *M.shape, format="csr")
    # M = Q * user_mat
    # print('matrix conversion finished !')
    #
    # per_user_idx_dict = {}
    # start = time.time()
    # for i in range(user_mat.shape[0]):
    #     indices = []
    #     for j in range(len(damping_factors)):
    #         # t1 = time.time()
    #         # tolerance 가 1.0e-3 or 1.0e-4 이면 indices list 값에 차이가 없음;
    #         # 최소 1.0e-5 이상 되어야 ...
    #         # idx = pagerank_scipy(M, N, target=i, alpha=damping_factors[j], tol=1.0e-3)
    #         idx = pagerank_scipy(M, N, target=i, alpha=damping_factors[j], tol=1.0e-5)
    #         indices.append(idx)
    #     per_user_idx_dict[i] = indices
    #     if i % 100 == 0:
    #         print('%d nodes processed!' % i)
    #         print('upto now %f S passed' % (time.time() - start))
    # print('calculation finished, now dumping the results !')
    # with open(os.path.join(data_dir, 'per_user_idx.dict'), 'wb') as f:
    #     pickle.dump(per_user_idx_dict, f)

    # item PageRank : gowalla = 40,981, yelp = 38,048, amazon = 91,599
    print('dataset: ', data_dir)
    item_mat = sp.sparse.load_npz(data_dir + '/item_mat.npz')
    print('item_mat : ', sys.getsizeof(item_mat))
    item_pd = pd.read_csv(data_dir + '/item_list.txt', sep=' ')
    nodelist = item_pd['remap_id'].to_list()
    M = item_mat
    # SHOULDn't user .toarray() -> wastes too much memory !!
    # also DON't need to conver the item_mat to todense() !!
    # use the original sparse matirx format !!!
    print('M : ', sys.getsizeof(M))
    N = len(nodelist)
    S = np.array(M.sum(axis=1)).flatten()
    print('S : ', sys.getsizeof(S))
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.spdiags(S.T, 0, *M.shape, format="csr")
    print('Q : ', sys.getsizeof(Q))
    M = Q * M
    print('M : ', sys.getsizeof(M))
    print('matrix conversion finished !')

    per_user_idx_dict = {}
    start = time.time()
    for i in range(item_mat.shape[0]):
        indices = []
        for j in range(len(damping_factors)):
            # t1 = time.time()
            # idx = pagerank_scipy(M, N, target=i, alpha=damping_factors[j], tol=1.0e-3)
            idx = pagerank_scipy(M, N, target=i, alpha=damping_factors[j], tol=1.0e-5)
            indices.append(idx)
        per_user_idx_dict[i] = indices
        if i % 100 == 0:
            print('%d nodes processed!' % i)
            print('upto now %f S passed' % (time.time() - start))
    print('calculation finished, now dumping the results !')
    with open(os.path.join(data_dir, 'per_item_idx.dict'), 'wb') as f:
        pickle.dump(per_user_idx_dict, f)
