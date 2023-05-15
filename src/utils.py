import numpy as np
def balanceData(X, y, k, p=0.5):
    mask = (y == k)
    not_msk = (y != k)

    X_isk = X[mask]

    X_notk = X[not_msk]

    totalk = mask.sum()
    total_notk = max(int(totalk * p),2)

    notk_ix = np.random.choice(X_notk.shape[0], total_notk)
    X_notk = X_notk[notk_ix]

    balanced_X = np.vstack([X_isk, X_notk])
    balanced_y = np.hstack([np.ones(X_isk.shape[0]), np.zeros(X_notk.shape[0])])

    return balanced_X, balanced_y


def clip_for_log(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1-eps)