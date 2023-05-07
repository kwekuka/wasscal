import sys
sys.path.append('../src/')
sys.path.append('../data/')

import numpy as np
from calibrate import *
from wasscal import *
from read_logits import *
from pycalib.metrics import ECE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# #Load data from CIFAR
# digits = load_digits() #from sklearn.datasets import load_digits
# X, y = digits.data, digits.target
#
# #Hyperparameters
#
# #for approximating distributions probabilities (Pr[Pk])
# n_bins = 1000
# bins = np.linspace(0,1,n_bins+1)
#
# #for visualizing reliability diagrams
# n_conf = 8
#
# #train test split param
# split_p = 0.5
#
# #Set length of dataset
# N = len(X)
# train_N = int(N*split_p)
# test_N = N - train_N
#
# #Number of classes
# K = 10
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     shuffle=False,
#     test_size=test_N,
# )
#
# lr = LogisticRegression(max_iter=2)
# lr.fit(X_train, y_train)
#
# y_uncal = lr.predict_proba(X_test)
#Load data from CIFAR
cifar10 = Loader("CIFAR100")
logits, simplex, labels = cifar10.logits, cifar10.simplex, cifar10.labels

#Hyperparameters

#for approximating distributions probabilities (Pr[Pk])
n_bins = 1000
bins = np.linspace(0,1,n_bins+1)

#for visualizing reliability diagrams
n_conf = 8

#train test split param
split_p = 0.5

#Set length of dataset
N = len(simplex)
train_N = int(N*split_p)
test_N = N - train_N

#Number of classes
K = simplex.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    logits,
    labels,
    shuffle=False,
    test_size=test_N,
)


calibrator = CalibrationLayer(method="Temperature", logits=X_train, labels=y_train)
ts = calibrator.calibrate().evaluate(X_test)
print(ECE(y_true = y_test, probs=ts)*100)

calibrator2 = CalibrationLayer(method="Vector", logits=X_train, labels=y_train, K=K)
vs = calibrator2.calibrate().evaluate(X_test)
print(ECE(y_true = y_test, probs=vs)*100)


calibrator3 = CalibrationLayer(method="Matrix", logits=X_train, labels=y_train, K=K)
ms = calibrator3.calibrate(ODIR=True, L2=True).evaluate(X_test)
print(ECE(y_true = y_test, probs=ms)*100)




X_train, X_test, y_train, y_test = train_test_split(
    simplex,
    labels,
    shuffle=False,
    test_size=test_N,
)


calibrator4 = CalibrationLayer(method="Dirichlet", logits=X_train, labels=y_train, K=K)
dc = calibrator4.calibrate(ODIR=True, L2=True).evaluate(X_test)
print(ECE(y_true = y_test, probs=dc)*100)

collect = []
for k in range(K):

    num_k = (y_train == k).sum()
    Xk_train, yk_train = X_train[y_train == k], np.ones(num_k)
    Xnotk_train, ynotk_train = X_train[y_train != k][:num_k], np.zeros(num_k)

    Xk_train = np.vstack([Xk_train, Xnotk_train])
    Yk_train = np.hstack([yk_train, ynotk_train])

    binaryCLF = RandomForestClassifier()
    binaryCLF.fit(Xk_train, Yk_train)
    class_k = binaryCLF.predict(X_test).astype(int)
    yk_true = (y_test == k).astype(int)
    yk_pred = X_test[:, k]
    bin_pred = snap(yk_pred, bins=bins).reshape(-1, 1)

    transported = bin_pred.copy()

    for kk in range(2):
        k_plan = getTransortPlanK(scores=bin_pred, y=class_k, k=kk, bins=bins)
        transported = applyTransportPlan(a=transported, M=k_plan, y=class_k, k=kk, bins=bins)

    collect.append(transported)

collected = np.hstack(collect)
collected = np.divide(collected, collected.sum(axis=1).reshape(-1, 1))
print(ECE(y_true = y_test, probs=collected)*100)

print(ECE(y_true = y_test, probs=X_test)*100)



#
# N = 5000
# n_bins = 10
# split_p = 0.5
# K = 2
# train_N = int(N*split_p)
# test_N = N - train_N
#
# X, y = make_classification(
#     n_samples=N, n_features=20, n_informative=2, n_redundant=2, random_state=42
# )
#
# train_samples = train_N  # Samples used for training the models
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     shuffle=False,
#     test_size=test_N,
# )
#
# # Create classifiers
# lr = LogisticRegression()
# gnb = GaussianNB()
# rfc = RandomForestClassifier()
#
# clf_list = [
#     (lr, "Logistic"),
#     # (gnb, "Naive Bayes"),
#     # (rfc, "Random forest"),
# ]
#
# bins = np.linspace(0,1,n_bins+1)
#
# for i, (clf, name) in enumerate(clf_list):
#     clf.fit(X_train, y_train)
#     cal_dists = []
#
#     # Get the probability of belonging to the Y=1 class
#     pos_pred = clf.predict_proba(X_test)[:, 1]
#
#     # snap scores to bins
#     bin_pred = snap(pos_pred, bins=bins).reshape(-1, 1)
#
#     # _, p_count = np.unique(bin_pred, return_counts=True)
#     # p_freq = p_count / np.sum(p_count)
#     #
#     # clb_neg = calibratedConditional(P=p_freq, y=y_test, k=0, bins=bins)
#     # clb_pos = calibratedConditional(P=p_freq, y=y_test, k=1, bins=bins)
#     #
#     # pmf_pos = binPMF(bin_pred, y_test, bins, k=1)
#     # pmf_neg = binPMF(bin_pred, y_test, bins, k=0)
#     #
#     # pos_plan = sinkhornTransport(pmf_pos, clb_pos, bins=bins)
#     # neg_plan = sinkhornTransport(pmf_neg, clb_neg, bins=bins)
#     #
#     y_pred = clf.predict(X_test)
#
#     transportPlans = getKTransportPlans(scores=bin_pred, y=y_pred, K=2, bins=bins)
#     transported = bin_pred.copy()
#     for k in range(K):
#         transported = applyTransportPlan(a=transported, M=transportPlans[k], y=y_pred, k=k, bins=bins)
#
