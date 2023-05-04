import numpy as np
from wasscal import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

N = 5000
n_bins = 10
split_p = 0.5
K = 2
train_N = int(N*split_p)
test_N = N - train_N

X, y = make_classification(
    n_samples=N, n_features=20, n_informative=2, n_redundant=2, random_state=42
)

train_samples = train_N  # Samples used for training the models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle=False,
    test_size=test_N,
)

# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
rfc = RandomForestClassifier()

clf_list = [
    (lr, "Logistic"),
    # (gnb, "Naive Bayes"),
    # (rfc, "Random forest"),
]

bins = np.linspace(0,1,n_bins+1)

for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    cal_dists = []

    # Get the probability of belonging to the Y=1 class
    pos_pred = clf.predict_proba(X_test)[:, 1]

    # snap scores to bins
    bin_pred = snap(pos_pred, bins=bins).reshape(-1, 1)

    # _, p_count = np.unique(bin_pred, return_counts=True)
    # p_freq = p_count / np.sum(p_count)
    #
    # clb_neg = calibratedConditional(P=p_freq, y=y_test, k=0, bins=bins)
    # clb_pos = calibratedConditional(P=p_freq, y=y_test, k=1, bins=bins)
    #
    # pmf_pos = binPMF(bin_pred, y_test, bins, k=1)
    # pmf_neg = binPMF(bin_pred, y_test, bins, k=0)
    #
    # pos_plan = sinkhornTransport(pmf_pos, clb_pos, bins=bins)
    # neg_plan = sinkhornTransport(pmf_neg, clb_neg, bins=bins)
    #
    y_pred = clf.predict(X_test)

    transportPlans = getKTransportPlans(scores=bin_pred, y=y_pred, K=2, bins=bins)
    transported = bin_pred.copy()
    for k in range(K):
        transported = applyTransportPlan(a=transported, M=transportPlans[k], y=y_pred, k=k, bins=bins)

