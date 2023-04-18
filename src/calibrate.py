import ot
import numpy as np
import functools


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
    """

    :param X: scores to make pmf
    :param y: class labels
    :param bins: possible bins
    :param k: class for which the binning should occur
    :return:
    """

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

def generateScores(count,bins):
    scores = []
    for i, c in enumerate(count):
        if c > 0:
            scores.append(np.ones(c)*bins[i])
    return np.hstack(scores).flatten()

def distributeMass(count, mass):
    #do the simple rounding solution
    distr = np.round(mass * count).astype(int)

    while distr.sum() < count:
        greedy_distr = (distr + (distr > 0).astype(int))/count

        #Get the index of the element that after having 1 added, would be closest to its true value
        resid_values = np.divide(greedy_distr, mass)
        resid = np.nanargmin(np.abs(1 - resid_values))
        distr[resid] += 1
    while distr.sum() > count:
        resid = np.argmax(distr/count - mass)
        distr[resid] -= 1

    return distr.astype(int)


def getKTransportPlans(scores, y, K, bins):
    plans = []
    freq = binPMF(scores, y, bins)
    for k in range(K):
        pmf = binPMF(scores, y, bins, k=k)
        clb = calibratedConditional(P=freq, y=y, k=k, bins=bins)
        plan = sinkhornTransport(pmf, clb, bins=bins)
        plans.append(plan)
    return plans

def applyTransportPlan(a, M, y, k, bins):
    new_scores = a.copy()
    class_mask = (y == k).reshape(-1,1)
    for i, bin in enumerate(bins):
        #apply transport only if there is mass there
        if M[i].sum() > 0:
            #Get the transport plan for the ith row, normalize
            ai = M[i]/np.sum(M[i])

            #Find all scores that belong to the ith row
            #i.e. the ith bin
            bin_mask = (a == bin)

            #Combined mask
            mask = np.logical_and(bin_mask, class_mask)

            #Determine how the number of objects in that bun
            count_val = np.sum(mask)

            # grid = np.linspace(0,1,count_val+1)
            snap_grid = distributeMass(count_val, ai)

            #Generate the scores according to the transport plan
            applyT = generateScores(snap_grid, bins)

            #Just to be sure I'm replacing the exact number of values that need replacing
            assert count_val == applyT.shape[0]

            #Replace old scores with new scores
            new_scores[mask] = applyT

    return new_scores




#
# def ot_scores_dist(a, b, binned, count, bins, labels, k):
#     plan = ot.emd_1d(bins, bins, a, b)
#     """
#     the plan is a 2d matrix so M[i][j] is the amount of mass in ai going to bj
#     """
#
#     mask = labels != k
#     y_new = binned.copy()[mask]
#     y_lab = labels.copy()[mask]
#     for i, X in enumerate(plan):
#         X = X/np.sum(X)
#         total_in_bin = count[i]
#         total_transport = np.ceil(np.multiply(total_in_bin, X)).astype(np.int)
#         transported_scores = make_transported_scores(total_transport,bins)
#         y_new = np.hstack([y_new, transported_scores])
#         y_lab = np.hstack([y_lab, np.array([k] * transported_scores.shape[0])])
#     return y_new, y_lab
#
#








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


# def monotone_transport(source, target):
#     SL = source + np.random.uniform(size=len(source)) * 0.01
#     TL = target + np.random.uniform(size=len(target)) * 0.01
#
#     SQ = np.sum(SL > SL.reshape([-1, 1]), axis=0).reshape(-1,1) / SL.shape[0]  # length 100 (for grid)
#     TQ = np.sum(TL > TL.reshape([-1, 1]), axis=0).reshape(-1,1) / TL.shape[0]  # length 100 (for grid)
#
#     # calculate best t index for each prediction to correct
#     ts_best = target[np.argmin(np.abs(SQ - TQ.T), axis=1)]
#
#     # return score adjustment
#     return ts_best - source

# def calibration_dist(freq, y_true, bins, k=1):
#     if k == 1:
#         cal_bin = np.multiply(freq,bins)
#     elif k == 0:
#         cal_bin = np.multiply(freq, 1-bins)
#     prY = np.mean(y_true == k)
#     clb_dist = cal_bin / prY
#     return clb_dist
#
# def calibration_projection(y_pred, y_true, bins, n=None):
#     if n is None:
#         n = y_pred.shape[0]
#     _, freq = np.unique(y_pred, return_counts=True)
#     freq = freq/freq.sum(axis=0)
#
#     clb_dist = np.multiply(bins, freq)/np.mean(y_true)
#     clb_dist = clb_dist/clb_dist.sum()
#
#     clb_scores = np.random.choice(bins, n, p=clb_dist)
#
#     return clb_scores
#
#
#
