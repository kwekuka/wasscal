import os
import pickle
from scipy.special import softmax


class Loader:
    """
    Loads logits and labels from a pickle file.
    Logits were saved from pretrained models defined here:
    https://github.com/chenyaofo/pytorch-cifar-models
    datasets: 'cifar10', 'cifar100'
    models: 'mobilenetv2_x0_5', 'mobilenetv2_x0_75', 'mobilenetv2_x1_0',
    'mobilenetv2_x1_4', 'repvgg_a0','repvgg_a1','repvgg_a2','resnet20', 'resnet32',
    'resnet44', 'resnet56', 'shufflenetv2_x0_5', 'shufflenetv2_x1_0',
    'shufflenetv2_x1_5', 'shufflenetv2_x2_0', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn',
    'vgg19_bn'
    probs: if True, computes the softmax of the logits in self.simplex
    """

    def __init__(self, dataset="cifar10", model="resnet20", probs=True):
        abs_path = os.path.abspath(__file__)
        dir_name = os.path.join(os.path.dirname(abs_path), "pretrained_logits")
        os.chdir(dir_name)

        # find verify that logits file exists
        f_name = "logits_" + dataset + "_" + model + "_pretrainedcy.p"
        if os.path.isfile(f_name):
            with open(f_name, "rb") as openfile:
                x = pickle.load(openfile)

            if probs:
                self.simplex = softmax(x[0], axis=1)
            self.logits = x[0]
            self.labels = x[1]
        else:
            raise FileNotFoundError("logits file not found")


# if __name__ == "__main__":
#    loader = Loader()
#    print(loader.logits.shape)
#    print(loader.labels.shape)
#    print(loader.simplex.shape)
