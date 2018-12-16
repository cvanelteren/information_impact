#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:14:16 2018

@author: casper
"""

from matplotlib.pyplot import *
from numpy import *
from scipy import optimize
import IO, os


def singleExp(x, a, b, c):
    return a + b * exp(-c * x)

def fitter(y, x, func, method):
    coefficients, co_var = optimize.curve_fit(func, x, y)
    
    return coefficients
    
def fit(y, func, 
        method = 'linearmixing'):
    
    xx = arange(y.shape[0])
    
    from multiprocessing import Pool, cpu_count
    from functools import partial
    tmpFunc = partial(fitter, x = xx, func = func, method = method)
    
    print(optimize.curve_fit(func, xx, y[:, 0]))
    with Pool(processes = cpu_count()) as p:
        results = p.map(tmpFunc, y.T)
    return results

def estimateIDT(func, coeffs, thetas, method):
    results = []
    for theta in thetas:
        theta = coeffs[0] 
        tmpFunc = lambda x : func(x, *coeffs) - theta    
        root = optimize.root(tmpFunc, x0 = 0, method = method)
if __name__ == '__main__':
    dataPath = f"{os.getcwd()}/Data/"
    tmp = "1539187363.9923286"
    
    data = IO.extractData(f"{dataPath}/{tmp}")
    t = next(iter(data)) # first
    samples  = data[t]['{}']
    
    func   = singleExp
    thetas = [10**-i for i in range(1, 20)]
    
    sampleResults = []
    for sample in samples:
        this = []
        x = arange(sample.mi.shape[0])
        for node_mi in sample.mi.T:
            thetaRoots = zeros(len(thetas))
            for idx, theta in enumerate(thetas):
                coeffs, _ = optimize.curve_fit(func, x, node_mi, maxfev = 10000)
                l       = coeffs[0] + theta * max(node_mi)
                root    = -1/coeffs[-1] * log((l - coeffs[0])/ coeffs[1]) # tmp
                
#                root    = optimize.root(lambda x : func(x, *coeffs) - l, 0, method = 'linearmixing').x
                root    = 0 if root < 0 or isnan(root) else root
                thetaRoots[idx] = root
            this.append(thetaRoots)
        sampleResults.append(array(this).argmax(axis = 0))
    sampleResults = array(sampleResults)
    consit = [max(histogram(i, arange(10), normed = 1)[0]) for  i in sampleResults.T]
    # %%
    fig, ax = subplots()
    ax.plot(thetas, consit, '--.', )
    setp(ax, **dict(xlabel = r'$\theta $', ylabel = 'Consistency', xscale = 'log'))
    
            
    
    
    
    
    
    

