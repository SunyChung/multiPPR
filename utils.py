import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


def RECALL(predictions, targets, k):
    topk_idx = torch.argsort(predictions, descending=True)[:k].detach().to('cpu').numpy()
    target_ids = np.where(targets == 1)[0]
    tmp = len(set(topk_idx) & set(target_ids))
    dinorm = min(k, targets.sum())
    return tmp / dinorm

def NDCG(predictions, targets, k):
    topk_idx = torch.argsort(predictions, descending=True)[:k].detach().to('cpu').numpy()
    discount = 1. / np.log2(np.arange(2, k+2))
    DCG = np.array(targets[topk_idx] * discount).sum()
    IDCG = discount[:min(targets.sum(dtype=np.int32), k)].sum()
    return DCG / IDCG


def result_plot(epochs, results, plot_label, y_label, save_name, title_label):
    params = {'legend.fontsize': 'small',
              'figure.figsize': (3.5, 2.5),
              'axes.labelsize': 'x-small',
              'axes.titlesize': 'x-small',
              'xtick.labelsize': 'xx-small',
              'ytick.labelsize': 'xx-small'}
    plt.rcParams['figure.constrained_layout.use'] = True
    pylab.rcParams.update(params)
    plt.figure()
    plt.plot(epochs, results, label=plot_label)
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    plt.title(title_label)
    plt.savefig(save_name)
    plt.show()
