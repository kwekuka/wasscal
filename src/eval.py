import sys
sys.path.append('../src/')
sys.path.append('../data/')
sys.path.append('../dirichlet/')

import numpy as np
from utils import *
from wasscal import *
from calibrate import *
from dataloader import uci, cifar
from scipy.special import softmax

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator

from sklearn.model_selection import (StratifiedKFold, GridSearchCV)


class EmptyEstimator(BaseEstimator, RegressorMixin):
    def fit(self, K):
        self.X_ = None
        self.y_ = None

        self.classes_ = np.arange(0, K)
        # Return the classifier
        return self

    def predict(self, X):
        return np.argmax(X, axis=1)

    def predict_proba(self, X):
        return X


class PytorchEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, method, lr=None, K=None):
        self.method = method
        self.clf = None
        self.lr = lr
        self.K = K

    def fit(self, X, y, **kwargs):
        self.X_ = X
        self.y_ = y

        if self.method == "Temperature":
            self.clf = CalibrationLayer(method="Temperature",
                                        logits=self.X_,
                                        labels=self.y_,
                                        lr=self.lr)
            self.clf.calibrate()

        elif self.method == "Vector":
            self.clf = CalibrationLayer(method="Vector",
                                        logits=self.X_,
                                        labels=self.y_,
                                        lr=self.lr,
                                        K=self.K
                                     )
            self.clf.calibrate()

    def predict_proba(self, X):
        return self.clf.evaluate(X)

    def predict(self, X):
        return np.argmin(self.clf.evaluate(X), axis=1)



class BenchmarkDataset:
    def __init__(self, dataset, model=None):
        if dataset in uci.uci_data:
            self.dataset = uci(dataset)
            X = self.dataset.data
            y = self.dataset.labels
            X_val, X_test, y_val, y_test = train_test_split(
                X,
                y,
                shuffle=True,
                test_size=int(X.shape[0] / 2),
            )

            self.X_val = X_val
            self.X_test = X_test
            self.y_val = y_val
            self.y_test = y_test

            self.clf = RandomForestClassifier()
            self.clf.fit(self.X_val, self.y_val)

            self.y_val_pred = self.clf.predict_proba(self.X_val)
            self.y_val_logits = clip_for_log(self.y_val_pred).astype(np.float32)

            self.y_test_pred = self.clf.predict_proba(self.X_test)
            self.y_test_logits = clip_for_log(self.y_test_pred).astype(np.float32)

            self.K = self.y_test_pred.shape[1]

        elif "cifar" in dataset:
            self.dataset = cifar(dataset)

            y_val_logits, y_test_logits, y_val, y_test = train_test_split(
                self.dataset.logits,
                self.dataset.labels,
                shuffle=True,
                test_size=int(self.dataset.logits.shape[0] * 3/ 4),
            )

            self.y_val = y_val
            self.y_test = y_test

            self.y_val_logits = y_val_logits
            self.y_test_logits = y_test_logits

            self.y_test_pred = softmax(self.y_test_logits, axis=1)
            self.y_val_pred = softmax(self.y_val_logits, axis=1)

            self.K = self.y_test_pred.shape[1]
        else:
            assert "dataset not supported"

class PlattScaling:
    def __init__(self, K):
        self.K = K
        self.platt = EmptyEstimator()
        self.platt.fit(self.K)
        self.clf = CalibratedClassifierCV(self.platt, cv="prefit", method="sigmoid")
    def fit(self, X, y):
        self.clf.fit(X, y)

    def calibrate(self, X):
        return self.clf.predict_proba(X)

    def ECE(self, input, labels):
        return ECE(self.calibrate(input), labels)

    def logits(self, X):
        return np.log(clip_for_log(self.calibrate(X)))
class IsotonicCalibration:
    def __init__(self, K):
        self.K = K
        self.iso = EmptyEstimator()
        self.iso.fit(self.K)
        self.clf = CalibratedClassifierCV(self.iso, cv="prefit", method="isotonic")
    def fit(self, X, y):
        self.clf.fit(X, y)

    def calibrate(self, X):
        return self.clf.predict_proba(X)

    def ECE(self, input, labels):
        return ECE(self.calibrate(input), labels)

    def logits(self, X):
        return clip_for_log(self.calibrate(X))

class TemperatureScaling:
    def __init__(self, learning_rates=None):
        self.temp = PytorchEstimator("Temperature")

        if learning_rates is None:
            self.lr = [1e-1, 1e-2, 1e-3, 1e-4]
        else:
            self.lr = learning_rates

        cv = StratifiedKFold(n_splits=max(len(self.lr),2), shuffle=True, random_state=0)


        self.clf = GridSearchCV(self.temp,
                                param_grid={'lr': self.lr},
                                cv=cv,
                                scoring='neg_log_loss')
    def fit(self, X, y):
        self.clf.fit(X,y)

    def calibrate(self, X):
        return self.clf.predict_proba(X)

    def ECE(self, input, labels):
        return ECE(self.calibrate(input), labels)

    def logits(self, X):
        return np.log(clip_for_log(self.calibrate(X)))

class VectorScaling:
    def __init__(self, K, learning_rates=None):
        self.K = K

        if learning_rates is None:
            self.lr = [1e-1, 1e-2, 1e-3, 1e-4]
        else:
            self.lr = learning_rates

        self.vec = PytorchEstimator(method="Vector", lr=self.lr, K=K)
        cv = StratifiedKFold(n_splits=max(len(self.lr),2), shuffle=True, random_state=0)

        self.clf = GridSearchCV(self.vec,
                                param_grid={'lr': self.lr},
                                cv=cv,
                                scoring='neg_log_loss')
    def fit(self, X, y):
        self.clf.fit(X,y)

    def calibrate(self, X):
        return self.clf.predict_proba(X)

    def ECE(self, input, labels):
        return ECE(self.calibrate(input), labels)

    def logits(self, X):
        return np.log(clip_for_log(self.calibrate(X)))

class DirichletCalibration:
    def __init__(self, reg=None):

        if reg is None:
            self.reg = [1e-2, 1e-3, 1e-4, 1e-5]
        else:
            self.reg = reg

        self.dc = FullDirichletCalibrator()
        cv = StratifiedKFold(n_splits=max(len(self.reg),2), shuffle=True, random_state=0)

        self.clf = GridSearchCV(self.dc,
                                param_grid={'reg_lambda': self.reg, 'reg_mu': [None]},
                                cv=cv,
                                scoring='neg_log_loss')
    def fit(self, X, y):
        self.clf.fit(X,y)

    def calibrate(self, X):
        return self.clf.predict_proba(X)

    def logits(self, X):
        return np.log(clip_for_log(self.calibrate(X)))

    def ECE(self, input, labels):
        return ECE(self.calibrate(input), labels)

class WassersteinCalibrator():
    def __init__(self, K, grain):
        self.K = K
        self.grid = np.linspace(0, 1, int(1/grain) + 1)

    # def fit(self, y_val_pred, y_val):
    #     self.plans = []
    #     self.k_clfs = []
    #
    #     for k in range(self.K):
    #
    #         # Cross validation for binary clfs
    #         acc = []
    #         clfs = []
    #         options = [0.05, .1, 0.25, 0.5]
    #         kf = KFold(n_splits=len(options))
    #         for i, (train_index, test_index) in enumerate(kf.split(y_val_pred)):
    #             Xk_train, Yk_train = balanceData(y_val_pred[train_index],
    #                                              y_val[train_index],
    #                                              k,
    #                                              p=options[i])
    #             binaryCLF = RandomForestClassifier()
    #             binaryCLF.fit(Xk_train, Yk_train)
    #             val_pk = binaryCLF.predict(y_val_pred[test_index]).astype(int)
    #             val_yk = (y_val == k)[test_index]
    #
    #             # collect potential classifiers
    #             acc.append(accuracy_score(y_true=val_yk, y_pred=val_pk))
    #             clfs.append(binaryCLF)
    #
    #         acc_max_ix = np.argmax(np.array(acc))
    #         clf = clfs[acc_max_ix]
    #
    #         k_plans = []
    #         is_pred_k = clf.predict(y_val_pred)
    #         yk_pred = y_val_pred[:, k]
    #         bin_pred = snap(yk_pred, bins=self.grid).reshape(-1, 1)
    #         for kk in range(2):
    #             k_plan = getTransortPlanK(scores=bin_pred, y=is_pred_k, k=kk, bins=self.grid)
    #             k_plans.append(k_plan)
    #
    #         self.plans.append(k_plans)
    #         self.k_clfs.append(clf)

    def calibrate(self, Y_train_pred, Y_test_pred, y_train, bins, K):
        return wassersteinCalibration(Y_train_pred,
                                      Y_test_pred,
                                      y_train,
                                      bins,
                                      K=K)








#
# for dataset in ["beans", "yeast"]:
#     bm = BenchmarkDataset(dataset=dataset)
#     w = wassersteinCalibration(bm.y_val_pred, bm.y_test_pred, bm.y_val, bins=np.linspace(0,1,101), K=bm.K)
#     print(ECE(w, bm.y_test))


