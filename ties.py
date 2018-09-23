from copy import copy
import networkx as nx

def computeStrengthOfTie(vi, vj, G):
    overlap                  = computeOverlap(vi, vj, G)
    # TODO: changes this; used for test case crash, i.e. no weighted edges
    try:
        weight = G[vi][vj]['weight']
    except:
        weight = 1
    strength                 = overlap * weight
    return strength

def computeOverlap(vi, vj, G):
    """
    Returns ratio of overlap of node i and node j
    """
    viNeighbors, vjNeighbors = copy(list(G.neighbors(vi))), copy(list(G.neighbors(vj))) 
    ki, kj                   = G.degree(vi), G.degree(vj)
    overlap = len(set(viNeighbors).intersection(vjNeighbors)) 
    return overlap / (ki + kj - 2 - overlap)
    

def computeStrengthOfTies(G):
    res = []
#    assert len(G.edges()) > 0
    for (vi, vj) in G.edges():
#        print(edge)
        res.append(computeStrengthOfTie(vi, vj, G))
#        res.append(computeStrengthOfTie(edge, G))
    return -1 if res == [] else res

def updateGraph(G):
    for (vi,vj) in G.edges():
#    G[vi][vj]['weight'] =  random.rand()
        G[vi][vj]['strength'] = computeStrengthOfTie(vi, vj, G)
        G[vi][vj]['overlap']  = computeOverlap(vi, vj, G)
    return G
#def checkForbidden(G):
#    for (i, j) in G.edges():
        


def fac(x):
    if x <= 1:
        return 1
    else:
        return fac(x-1) * x

def binom(x, n):
    return fac(x) / (fac(x-n) * fac(n))

