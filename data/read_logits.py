import pickle 
import os

if __name__== "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    with (open("logits_c10_pretrained.p", "rb")) as openfile:
        x = pickle.load(openfile)

    logits = x[0] #shape (#samples, #classes) (10000,10)
    labels = x[1] # shape (#samples) (10000,)
    print(logits.shape)
    print(labels.shape)