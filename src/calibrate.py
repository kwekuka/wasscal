import numpy as np
import ot


def snap(y, bins):
    """
    Snap the scores in y to the bins
    :param y: some list of scores that need to be snapped to the bins (next argument)
    :param bins: the set of possible scores that the y (var) will be mapped to
    :return: np.array with dim (y.shape[0] x 1)
    """

    #Get the number of people belonging to each bin, np.histogram method does this for you
    _, bin_edges = np.histogram(y, bins=bins)

    #Make the vectors we're dealing with have the right dimmension, this is a hygeine step
    y, bin_edges = y.reshape(-1,1), bin_edges.reshape(-1,1)

    #Compute the difference between the scores, and each of the possible scores
    bin_diff = np.abs(y - bin_edges.T)

    #Get the index of the smallest difference, and get the gorresponding bin
    snapped = bin_edges[np.argmin(bin_diff, axis=1)]

    return snapped.flatten()

def calibratedConditional(P, y, k, bins):
    """
    :param P:  The histogram representing the Pr(P=p)
    :param y:  The class labels (each element should be 0 ... |K|-1)
    :param k:  The class we'd like to calibrated (ranges from 0 .. |K| - 1)
    :param bins: The bins representing the different possible scores
    :return:  the calibrated distribution as histogram over bin edges

    Assumptions:
    1. Elements in P are already binned
    2. P.shape[0] = y.shape[0]
    """

    isBinary = (np.unique(y).shape[0] <= 2)

    if k == 0 and isBinary:
        bins = np.flip(bins)
    prior = np.multiply(bins, P)
    likelihood = 1/np.mean(y == k)

    return prior * likelihood

def binPMF(X, y, bins, k=None):

    if k is not None: 
        X = X[y == k]
    
    hist = np.array([np.sum(X == b) for b in bins])
    hist = hist/np.sum(hist)

    return hist
def sinkhornTransport(source, target, bins):
    """

    :param source:
    :param target:
    :param bins:
    :return: OT Plan as matrix

    G0 is a.shape[0] x b.shape[0] matrix where G0[i,j] is mass in a[i] to b[j]
    """
    G0 = ot.emd_1d(bins, bins, source, target)
    return G0

def applyTransportPlan(a, M, y, k, bins):
    new_scores = a.copy()
    class_mask = (y == k).reshape(-1,1)
    for i, bin in enumerate(bins):
        #Get the transport plan for the ith row, normalize
        ai = M[i]/np.sum(M[i])

        #Find all scores that belong to the ith row
        #i.e. the ith bin 
        bin_mask = (a == bin)

        #Combined mask
        mask = np.logical_and(bin_mask, class_mask)

        #Determine how the number of objects in that bun 
        count_val = np.sum(mask)

        #Generate the scores according to the transport plan  
        applyT = np.random.choice(bins, count_val, p=ai)

        #Replace old scores with new scores 
        new_scores[mask] = applyT
    return new_scores



def make_transported_scores(x, bins):
    transported = []
    for i, _ in enumerate(x):
        transported_scores = [bins[i]] * x[i]
        transported += transported_scores
    return np.array(transported).reshape(-1)

def ot_scores_dist(a, b, binned, count, bins, labels, k):
    plan = ot.emd_1d(bins, bins, a, b)
    """
    the plan is a 2d matrix so M[i][j] is the amount of mass in ai going to bj 
    """

    mask = labels != k
    y_new = binned.copy()[mask]
    y_lab = labels.copy()[mask]
    for i, X in enumerate(plan):
        X = X/np.sum(X)
        total_in_bin = count[i]
        total_transport = np.ceil(np.multiply(total_in_bin, X)).astype(np.int)
        transported_scores = make_transported_scores(total_transport,bins)
        y_new = np.hstack([y_new, transported_scores])
        y_lab = np.hstack([y_lab, np.array([k] * transported_scores.shape[0])])
    return y_new, y_lab










    # SL = source
    # # SL = source + np.random.uniform(size=len(source)) * 0.01
    # TQ = np.cumsum(target).reshape(-1,1)
    #
    # SQ = np.sum(SL > SL.reshape([-1, 1]), axis=0).reshape(-1,1) / SL.shape[0]  # length 100 (for grid)
    #
    # # calculate best t index for each prediction to correct
    # ts_best = bins[np.argmin(np.abs(SQ - TQ.T), axis=1)]
    #
    # # return score adjustment
    # return ts_best


def monotone_transport(source, target):
    SL = source + np.random.uniform(size=len(source)) * 0.01
    TL = target + np.random.uniform(size=len(target)) * 0.01

    SQ = np.sum(SL > SL.reshape([-1, 1]), axis=0).reshape(-1,1) / SL.shape[0]  # length 100 (for grid)
    TQ = np.sum(TL > TL.reshape([-1, 1]), axis=0).reshape(-1,1) / TL.shape[0]  # length 100 (for grid)

    # calculate best t index for each prediction to correct
    ts_best = target[np.argmin(np.abs(SQ - TQ.T), axis=1)]

    # return score adjustment
    return ts_best - source

def calibration_dist(freq, y_true, bins, k=1):
    if k == 1:
        cal_bin = np.multiply(freq,bins)
    elif k == 0:
        cal_bin = np.multiply(freq, 1-bins)
    prY = np.mean(y_true == k)
    clb_dist = cal_bin / prY
    return clb_dist

def calibration_projection(y_pred, y_true, bins, n=None):
    if n is None:
        n = y_pred.shape[0]
    _, freq = np.unique(y_pred, return_counts=True)
    freq = freq/freq.sum(axis=0)

    clb_dist = np.multiply(bins, freq)/np.mean(y_true)
    clb_dist = clb_dist/clb_dist.sum()

    clb_scores = np.random.choice(bins, n, p=clb_dist)

    return clb_scores



