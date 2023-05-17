import sys
sys.path.append('../src/')
sys.path.append('../data/')
sys.path.append('../dirichlet/')

import numpy as np
from utils import *
from sklearn import svm
from wasscal import *
from calibrate import *
from dataloader import uci, cifar
from scipy.special import softmax
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import calibration_curve
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
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
        self.temp = None


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
            X = self.dataset.data.astype(np.float32)
            y = self.dataset.labels.astype(np.int_)

            # X = normalize(X, axis=1)

            # unique, count = np.unique(y, return_counts=True)
            # too_small = count < 10
            # keep_rows = (y != unique[too_small].reshape(-1, 1)).any(axis=0)

            # X = X[keep_rows]
            # y = y[keep_rows]
            #
            # y

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
        print()

    def calibrate(self, X):
        return self.clf.predict_proba(X)

    def ECE(self, input, labels):
        return ECE(self.calibrate(input), labels)

    def logits(self, X):
        return clip_for_log(self.calibrate(X))

class TemperatureScaling:
    def __init__(self, K, learning_rates=None):
        self.K = K
        self.temp = PytorchEstimator("Temperature")

        if learning_rates is None:
            self.lr = [1e-1, 1e-2, 1e-3, 1e-4]
        else:
            self.lr = learning_rates

        cv = StratifiedKFold(n_splits=max(len(self.lr),2), shuffle=True, random_state=0)


        self.clf = GridSearchCV(self.temp,
                                param_grid={'lr': self.lr},
                                cv=cv,
                                scoring='neg_log_loss',
                                refit=True)

        print()

    def fit(self, X, y):
        self.clf.fit(X,y)
    def calibrate(self, X):
        return self.clf.predict_proba(X)

    def ECE(self, input, labels):
        return ECE(self.calibrate(input), labels)

    def logits(self, X):
        return np.log(clip_for_log(self.calibrate(X)))

    def predict(self, input):
        return self.clf.predict(input)

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

class BalancedBinary(BaseEstimator, RegressorMixin):
    def __init__(self, k, p=None):
        self.p = p
        self.k = k

    def balance(self, X, y):
        mask = (y == 1)
        not_msk = (y == 0)

        X_isk = X[mask]

        X_notk = X[not_msk]

        totalk = mask.sum()
        total_notk = max(int(totalk * self.p), 2)

        notk_ix = np.random.choice(X_notk.shape[0], total_notk)
        X_notk = X_notk[notk_ix]

        balanced_X = np.vstack([X_isk, X_notk])
        balanced_y = np.hstack([np.ones(X_isk.shape[0]), np.zeros(X_notk.shape[0])])

        return balanced_X, balanced_y
    def fit(self, X,y):

        self.X_, self.y_ = self.balance(X,y)
        self.classes_ = np.arange(0, 2)
        self.clf = MLPClassifier(max_iter=500, learning_rate='adaptive')
        # self.clf = RandomForestClassifier()
        self.clf.fit(self.X_, self.y_)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

class WassersteinCalibrator():
    def __init__(self, K,grain, cv=2):
        self.K = K
        self.plans = []
        self.k_clfs = []
        self.cv = cv
        self.grid = np.linspace(0, 1, int(1/grain) + 1)


    def getKClassifier(self, X, y, k, options=None):
        if options is None:
            options = [0.05, .1, 0.25, 0.5, .75]
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True)
        bal_bin = BalancedBinary(k=k)
        k_clf = GridSearchCV(bal_bin,
                             param_grid={'p': options},
                             cv=cv,
                             scoring='accuracy')

        k_clf.fit(X, y)
        return k_clf
    def fit(self, y_val_pred, y_val, y_test_pred):

        self.predictions = []
        self.k_clfs = []
        self.plans = []

        for k in range(self.K):
            kclf = self.getKClassifier(y_val_pred, (y_val == k).astype(int), k=k)
            self.k_clfs.append(kclf)
            self.predictions.append(kclf.predict(y_test_pred).reshape(-1,1))


        self.predictions = np.hstack(self.predictions)

        for k in range(self.K):

            is_pred_k = self.predictions[:,k]
            yk_pred = y_test_pred[:, k]
            bin_pred = snap(yk_pred, bins=self.grid).reshape(-1, 1)

            self.plans.append([
                getTransortPlanK(scores=bin_pred, y=is_pred_k, k=0, bins=self.grid),
                getTransortPlanK(scores=bin_pred, y=is_pred_k, k=1, bins=self.grid)
            ])



    def calibrate(self, Y_test_pred):
        collect = []

        for k in range(self.K):

            clf = self.k_clfs[k]
            is_pred_k = clf.predict(Y_test_pred)
            yk_pred = Y_test_pred[:, k]
            bin_pred = snap(yk_pred, bins=self.grid).reshape(-1, 1)

            transported = bin_pred.copy()

            for kk in range(2):
                k_plan = self.plans[k][kk]
                if k_plan is not None:
                    transported = applyTransportPlan(a=transported, M=k_plan, y=is_pred_k, k=kk, bins=self.grid)

            collect.append(transported)

        collected = np.hstack(collect)
        return np.divide(collected, collected.sum(axis=1).reshape(-1, 1))


    def logits(self, X):
        return np.log(clip_for_log(self.calibrate(X)))

    def ECE(self, input, labels):
        return ECE(self.calibrate(input), labels)


# for dataset in ["beans"]:
#
#     bm = BenchmarkDataset(dataset=dataset)
#     print(ECE(bm.y_test_pred, bm.y_test))
#
#     ts = TemperatureScaling(K=bm.K)
#     ts.fit(bm.y_val_logits, bm.y_val)
#     print("TS: %f " % ts.ECE(bm.y_test_logits, bm.y_test))
#
#
#     vs = VectorScaling(K=bm.K)
#     vs.fit(bm.y_val_logits, bm.y_val)
#     print("VS: %f " % vs.ECE(bm.y_test_logits, bm.y_test))
#
#
#     #Samples of fitting and computing calibration error
#     platt = PlattScaling(bm.K)
#     platt.fit(bm.y_val_pred, bm.y_val)
#     print("Platt: %f " % platt.ECE(bm.y_test_pred, bm.y_test))
#
#
#
#     iso = IsotonicCalibration(bm.K)
#     iso.fit(bm.y_val_pred, bm.y_val)
#     print("Iso: %f " % iso.ECE(bm.y_test_pred, bm.y_test))
#
#
#     # dc = DirichletCalibration()
#     # dc.fit(bm.y_val_logits, bm.y_val)
#     # print(dc.ECE(bm.y_test_logits, bm.y_test))
#
#     w = WassersteinCalibrator(K=bm.K, grain=0.05)
#     w.fit(bm.y_val_pred, bm.y_val, bm.y_test_pred)
#     print("Wasserstein: %f " % w.ECE(bm.y_test_pred, bm.y_test))






#
# for dataset in ["beans"]:
#     bm = BenchmarkDataset(dataset=dataset, model="vgg16_bn")
#
#
#     w = WassersteinCalibrator(K=bm.K, grain=0.01)
#     w.fit(bm.y_val_pred, bm.y_val, bm.y_test_pred)
#     print(w.ECE(bm.y_test_pred, bm.y_test))
#
#
#     ts = TemperatureScaling()
#     ts.fit(bm.y_val_logits, bm.y_val)
#     print(ts.ECE(bm.y_test_logits, bm.y_test))
#
#
#
#
#     #Samples of fitting and computing calibration error
#     platt = PlattScaling(bm.K)
#     platt.fit(bm.y_val_pred, bm.y_val)
#     print(platt.ECE(bm.y_test_pred, bm.y_test))
#
#     #Example of how to get calibrated logits or calibrated probabilities
#     platt.logits(bm.y_test_pred)
#     platt.calibrate(bm.y_test_pred)
#
#     iso = IsotonicCalibration(bm.K)
#     iso.fit(bm.y_val_pred, bm.y_val)
#     print(iso.ECE(bm.y_test_pred, bm.y_test))


    # dc = DirichletCalibration()
    # dc.fit(bm.y_val_logits, bm.y_val)
    # print(dc.ECE(bm.y_test_logits, bm.y_test))



    # vs = VectorScaling(K=bm.K)
    # vs.fit(bm.y_val_logits, bm.y_val)
    # print(vs.ECE(bm.y_test_logits, bm.y_test))
    #


# for dataset in ["abalone"]:
#     bm = BenchmarkDataset(dataset=dataset)
#
#     w = WassersteinCalibrator(K=bm.K, grain=0.001)
#     w.fit(bm.y_val_pred, bm.y_val, bm.y_test_pred)
#     hi = w.calibrate(bm.y_test_pred)
#     print(ECE(bm.y_test_pred, bm.y_test))
#     print(ECE(hi, bm.y_test))

#
#     # w = wassersteinCalibration(bm.y_val_pred, bm.y_test_pred, bm.y_val, bins=np.linspace(0,1,101), K=bm.K)
#
# #
# for dataset in ["cifar10", "beans", "yeast"]:
#     bm = BenchmarkDataset(dataset=dataset)
#
#     dc = DirichletCalibration()
#     dc.fit(bm.y_val_logits, bm.y_val)
#     print(dc.ECE(bm.y_test_logits, bm.y_test))
#
#     #Samples of fitting and computing calibration error
#     platt = PlattScaling(bm.K)
#     platt.fit(bm.y_val_pred, bm.y_val)
#     print(platt.ECE(bm.y_test_pred, bm.y_test))
#
#     #Example of how to get calibrated logits or calibrated probabilities
#     platt.logits(bm.y_test_pred)
#     platt.calibrate(bm.y_test_pred)
#
#     iso = IsotonicCalibration(bm.K)
#     iso.fit(bm.y_val_pred, bm.y_val)
#     print(iso.ECE(bm.y_test_pred, bm.y_test))
#
#
#
#     ts = TemperatureScaling()
#     ts.fit(bm.y_val_logits, bm.y_val)
#     print(ts.ECE(bm.y_test_logits, bm.y_test))
#
#     vs = VectorScaling(K=bm.K)
#     vs.fit(bm.y_val_logits, bm.y_val)
#     print(vs.ECE(bm.y_test_logits, bm.y_test))


#





# bm = BenchmarkDataset(dataset="abalone")
# clf = RandomForestClassifier()
# clf.fit(bm.X_val, bm.y_val)
#
# print(accuracy_score())