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
    return linalg.norm( (sqrt(p1) - sqrt(p2)), axis = -1) / sqrt(2)


def hdTime(x):
    return 0
