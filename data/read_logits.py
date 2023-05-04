import os
import pickle
import numpy as np
from scipy.special import softmax

class Loader():
    def __init__(self, dataset, probs=True):
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        if dataset == "CIFAR10":
            with (open("logits_c10_pretrained.p", "rb")) as openfile:
                x = pickle.load(openfile)
        elif dataset == "CIFAR100":
            with (open("logits_c100_pretrained.p", "rb")) as openfile:
                x = pickle.load(openfile)

        #
        if probs:
            self.simplex = softmax(x[0], axis=1)
        self.logits = x[0]
        self.labels = x[1]


# if __name__== "__main__":
#     abspath = os.path.abspath(__file__)
#     dname = os.path.dirname(abspath)
#     os.chdir(dname)
#     with (open("logits_c10_pretrained.p", "rb")) as openfile:
#         x = pickle.load(openfile)
#
#     logits = x[0] #shape (#samples, #classes) (10000,10)
#     labels = x[1] # shape (#samples) (10000,)
#     print(logits.shape)
#     print(labels.shape)