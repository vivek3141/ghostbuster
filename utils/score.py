import numpy as np

from sklearn.linear_model import LogisticRegression
from torch.utils.data import random_split


def k_fold_score(X, labels, indices=None, k=8, precision=10):
    if indices is None:
        indices = np.arange(X.shape[0])

    splits = [len(indices) // k] * k
    splits[-1] += len(indices) % k
    k_split = random_split(indices, splits)

    score_sum = 0
    for i in range(k):
        train = np.concatenate([np.array(k_split[j]) for j in range(k) if i != j])
        model = LogisticRegression(C=10, penalty="l2", max_iter=1000)
        model.fit(X[train], labels[train])

        score_sum += model.score(X[k_split[i]], labels[k_split[i]])

    return round(score_sum / k, precision)
