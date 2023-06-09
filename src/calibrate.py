import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassCalibrationError
from sklearn.isotonic import IsotonicRegression
from utils import *
import scipy


class MatrixScaling(nn.Module):
    def __init__(self, K):
        super(MatrixScaling, self).__init__()
        self.matrix = nn.Linear(K,K)

    def forward(self, input):
        return self.matrix(input)

    def evaluate(self, input):
        input = torch.tensor(input)
        with torch.no_grad():
            return torch.torch.nn.functional.softmax(self.forward(input), dim=1).numpy()

class DirichletCalibration(nn.Module):
    def __init__(self, K, eps=1e-12):
        super(DirichletCalibration, self).__init__()
        self.matrix = nn.Linear(K,K)
        self.eps = eps

    def forward(self, input):
        with torch.no_grad():
            logits = torch.log(torch.clip(input, self.eps, 1-self.eps))

        return self.matrix(logits)

    def evaluate(self, input):
        input = torch.tensor(input)
        with torch.no_grad():
            logits = torch.log(torch.clip(input, self.eps, 1 - self.eps))
            return torch.torch.nn.functional.softmax(self.forward(logits), dim=1).numpy()
class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.T = nn.Parameter(torch.ones(1),requires_grad=True)

    def forward(self, input):
        return self.temperature_scale(input)

    def evaluate(self, input):
        input = torch.tensor(input)
        with torch.no_grad():
            return torch.torch.nn.functional.softmax(self.forward(input), dim=1).numpy()

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        temperature = self.T
        return torch.div(logits, temperature)


class VectorScaling(nn.Module):
    def __init__(self, K):
        super(VectorScaling, self).__init__()
        self.affine = nn.Parameter(torch.ones(K), requires_grad=True)
        self.intercept = nn.Parameter(torch.zeros(K), requires_grad=True)



    def forward(self, input):
        return torch.mul(self.affine, input) + self.intercept

    def evaluate(self, input):
        input = torch.tensor(input)
        with torch.no_grad():
            return torch.torch.nn.functional.softmax(self.forward(input), dim=1).numpy()


class CalibrationLayer():
    def __init__(self, method, logits, labels, lr=1e-8, K=None):
        self.logits = torch.tensor(logits)
        self.labels = torch.tensor(labels)
        self.method = method
        self.K = K
        self.lr = lr

        if method == "Temperature":
            self.calibrator = TemperatureScaling()

        if method == "Vector":
            self.calibrator = VectorScaling(K=K)

        # if method == "Matrix":
        #     self.calibrator = MatrixScaling(K=K)
        #
        # if method == "Dirichlet":
        #     self.calibrator = DirichletCalibration(K=K)

    def evaluate(self, X):
        if self.method == "Temperatures":
            print(self.calibrator.par.detach().numpy())

        return self.calibrator.evaluate(X)

    def _calibrate(self):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS(self.calibrator.parameters(),
                                self.lr,
                                max_iter=500)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.calibrator(self.logits), self.labels)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        return self, loss.detach().flatten().numpy()

    def _calibrateODIRL2(self):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS(self.calibrator.parameters(), lr=self.lr, max_iter=500)

        r_odir = .5
        r_bias = .75
        k = self.K

        def odir_reg(matrix):
            sum = torch.tensor(0.0)
            for i, row in enumerate(matrix):
                for j, _ in enumerate(row):
                    if i != j:
                        sum += torch.pow(matrix[i][j],2)
            return sum


        def closure():
            optimizer.zero_grad()

            odir = self.calibrator.matrix.weight
            # odir.diagonal(dim1=-1, dim2=-2).zero_()

            bias = self.calibrator.matrix.bias.clone()

            loss = nll_criterion(self.calibrator(self.logits), self.labels)
            odir_loss = odir_reg(odir) * (1/(k*(k-1))) * r_odir
            bias_loss = torch.norm(bias, p=2) * (1/k) * r_bias
            loss += odir_loss
            loss += bias_loss
            loss.backward()
            print(ECE(probs=self.evaluate(self.d)))
            return loss

        loss = optimizer.step(closure)
        return self, loss.detach().flatten().numpy()
    def calibrate(self, ODIR=False, L2=False):
        if ODIR and L2:
            return self._calibrateODIRL2()
        else:
            return self._calibrate()





def ECE(probs, labels, bins=100):
    probs = np.divide(probs, probs.sum(axis=1).reshape(-1, 1))
    ce = MulticlassCalibrationError(num_classes=probs.shape[1], n_bins=bins, norm='l1')
    probs = torch.tensor(probs)
    labels = torch.tensor(labels)

    return ce(probs, labels)*100

