import numpy as np
'''Dump file of statistical related functions'''
def hellingerDistance(p1, p2):
    '''
    input:
        :p1: probability control
        :p2: probability of nudge conditions
    Note: it assumes the last axis contains the probability distributions to be compared
    returns:
        hellinger distance between p1, p2
    '''
    return np.linalg.norm( p1 - p2, ord = 2, axis = -1) / np.sqrt(2)

def KL(p1, p2):
    kl = - np.nansum(p1 * (np.log(p2) - np.log(p1)), axis = -1)
    kl[np.isfinite(kl) == False] = 0 # remove x = 0
    return kl

def KL2(p1, p2):
    from scipy.special import kl_div
    return np.nansum(kl_div(p1, p2), axis = -1)

def hdTime(x):
    return 0

def rankData(x):
    '''
    Assumes x has shape assumes nodes is the last dimension
    Returns the ranked data
    It is not the most efficient method used
    '''
    from scipy.stats.mstats import rankdata
    s        = x.shape
    ranking  = np.zeros(s)
    maxim    = np.zeros( s[:-1] )
    maxim[:] = np.nan
    
    # reshape (not really productive)
    maxim    = maxim.reshape( -1 )
    ranking  = ranking.reshape( (  -1, s[-1] ) )
    noneFound= 0
    mask = np.ma.masked_invalid(x)
    for idx, sample in enumerate(mask.reshape(-1, s[-1])):
        # allow only if there is variance in the data
        if sample.sum() != 0:
            rank         = rankdata(sample)
            ranking[idx] = rank
            maxim[idx]   = rank.argmax()
        else:
            noneFound += 0
    ranking = ranking.reshape(s) # reshape back
    maxim   = maxim.reshape(s[:-1])
    print(f'In {noneFound} trials no max are found')
    return ranking, maxim

def accuracy(rankings, target):
    """
    Produces the frequency of correctness for the ranking and the targets
    Input:
        :ranking: nSamples x predictors
        :target: nSamples x 1
    """
    return (rankings == target).mean(0)
    