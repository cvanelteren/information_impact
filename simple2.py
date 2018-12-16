#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:09:36 2018

@author: casper
"""
from numpy import *
from scipy import optimize, integrate
import IO, os, plotting as plotz
from matplotlib.pyplot import *
from time import sleep
style.use('seaborn-poster')
dataPath = f"{os.getcwd()}/Data/"
extractThis      = IO.newest(dataPath)[-1]
tmp      = '1539187363.9923286'
#tmp      = '1540135977.9857328'
extractThis = f"{dataPath}{tmp}"

data   = IO.extractData(extractThis)

# thetas = [10**-i for i in range(1, 20)]
thetas  = logspace(log10(.8), -25, 10)
#thetas  = array([.5, .1, .01, .001])
t       = next(iter(data))
model   = data[t]['{}'][0].model


deltas = IO.readSettings(extractThis)['deltas']
controls = array([i.mi for i in data[t]['{}']])
roots    = zeros((len(controls), model.nNodes, len(thetas), 2))

colors = cm.tab20(arange(model.nNodes))

dd = zeros((len(controls), model.nNodes, deltas // 2 + 1,  2))
for condition, samples in data[t].items():
    for idx, sample in enumerate(samples):
        if condition == '{}':
            dd[idx, ..., 0] = sample.mi[:deltas // 2 + 1, :].T
#            ax.plot(sample.mi)
        else:
            impact = plotz.hellingerDistance(sample.px, data[t]['{}'][idx].px).mean(-1)
            impact = impact[deltas // 2  : ][None, :].T
            # TODO: check if this works with tuples (not sure)
            jdx = [model.mapping[int(j)] if j.isdigit() else model.mapping[j]\
                for key in model.mapping for j in re.findall(str(key), re.sub(':(.*?)\}', '', condition))]
            dd[idx, jdx, ...,  1] = impact.squeeze().T
#ax.set_xlim(0, 4)        
# %%
func  = lambda x, a, b, c: a + b * exp(- c * x) #  + d * exp(-e * (x - f))
p0          = ones((func.__code__.co_argcount - 1)); # p0[0] = 0
fitParam    = dict(maxfev = int(1e3), bounds = (0, inf))

mus = dd.mean(0)


aucs  = zeros((model.nNodes, len(thetas), 2))
roots = zeros((model.nNodes, len(thetas), 2))

x = arange(deltas // 2)
fig, ax = subplots()
for idx in range(mus.shape[-1]):
    coeffs, errors = plotz.fit(mus[..., idx].T, func, params = fitParam)
    print(errors)
    for cidx, c in enumerate(coeffs):
        for tidx, theta in enumerate(thetas):
            m    = sum(c[:2])
            
            theta =  theta * (m - c[0]) + c[0] 
#            theta = theta * (c[0] + c[1.]) 
            tmpFunc = lambda x: abs(func(x, *c) - theta)
            
            rr = (- 1/c[2])* log( (theta - c[0] ) / c[1])#
#            root    = optimize.root(tmpFunc, 0, method = 'linearmixing')  
#            root = optimize.root(tmpFunc, rr, method = 'excitingmixing')
#            print(root, rr)
#            if root.success:
#                print('yes')
#            root = root.x 
            root = rr
                   
            root = 0 if root < 0 or isnan(root) else root
            xx   = linspace(0, root, 1000)
            auc =  integrate.simps(func(xx, *c), xx, even = 'avg')
            
            roots[cidx, tidx, idx] = root
            aucs[cidx, tidx, idx]  = 0 if isnan(auc) else auc
# %% time plots
fig, axes = subplots(2, sharex = 'all')
mainAx  = fig.add_subplot(111, frameon = False)
setp(mainAx, **dict(xticks = [], yticks = []))
mainAx.set_xlabel('time [step]', labelpad = 20)
for ax, mu in zip(axes.ravel(), mus.T):
    [ax.plot(mui, color = c) for mui, c in zip(mu.T, colors)]
    ax.set_yscale('log')
#    ax.set_xlim(0, 10)
labels = 'MI <H>'.split()
[axi.set_ylabel(label) for axi, label in zip(axes.ravel(), labels)]
mainAx.legend(handles = ax.lines, labels = [node for node in model.graph.nodes()], \
                        bbox_to_anchor = (1.20, 1), title = 'Node')
# %% area under the curve plots
fig, ax = subplots(2, sharex = 'all')
mainAx  = fig.add_subplot(111, frameon = False)
setp(mainAx, **dict(xticks = [], yticks = []))
for axi, i in zip(ax.ravel(), aucs.T):
    [axi.plot(thetas, ii, color = c) for idx, (ii, c) in enumerate(zip(i.T, colors))\
     ]
    axi.set_yscale('log')
    axi.set_xscale('log')
    axi.set_xlim(thetas[0], thetas[-1])
labels = r'$AUC_{MI}$ $AUC_{<H>}$'.split()
[axi.set_ylabel(label, labelpad = 10) for axi, label in zip(ax.ravel(), labels)]
mainAx.set_xlabel(r'$\theta$', labelpad = 20)
mainAx.legend(handles = axi.lines, labels = [node for node in model.graph.nodes()], \
                        bbox_to_anchor = (1.20, 1), title = 'Node')
#mainAx.axis('off')
# %% roots curves
fig, ax = subplots(2, sharex = 'all')
mainAx  = fig.add_subplot(111, frameon = False)
setp(mainAx, **dict(xticks = [], yticks = []))
for axi, i in zip(ax.ravel(), roots.T):
    [axi.plot(thetas, ii, color = c) for ii, c in zip(i.T, colors)]
    axi.set_yscale('log')
    axi.set_xscale('log')
    axi.set_xlim(thetas[0], thetas[-1])
labels = r'$root_{MI}$ $root_{<H>}$'.split()
[axi.set_ylabel(label, labelpad = 10) for axi, label in zip(ax.ravel(), labels)]
mainAx.set_xlabel(r'$\theta$', labelpad = 20)
mainAx.legend(handles = axi.lines, labels = [node for node in model.graph.nodes()], \
                        bbox_to_anchor = (1.20, 1), title = 'Node')

# %% consistency only works over multiple samples
from mpl_toolkits.mplot3d import Axes3D
fig, ax = subplots(subplot_kw = dict(projection = '3d'))
for c, x in zip(colors, roots):
    ax.scatter(*log10(x.T), zs = log10(thetas), color = c, alpha =1)
    
# %% concistency
rootsc = zeros((dd.shape[0], model.nNodes, len(thetas), 2))
for idx, sample in enumerate(dd):
    # node x time x 2
    for jdx, s in enumerate(sample.T):
        coeffs, errors = plotz.fit(s, func, params = fitParam)
        for kdx, c in enumerate(coeffs):
            m    = sum(c[:2])
            for zdx, theta in enumerate(thetas):
                theta =  theta * (m - c[0]) #  + c[0] 
                rr = (- 1/c[2])* log( (theta - c[0]) / c[1])
                rr = 0 if rr < 0 or isnan(rr) else rr
                rootsc[idx, kdx, zdx, jdx] = rr
    
 # %% compute concistency
cons = lambda x : max(histogram(x, arange(model.nNodes), density = True)[0])

maxim = rootsc.argmax(axis = 1)
consistency = array([ [cons(i) for i in j] for j in maxim.T])
tmp = maxim[..., 0] == maxim[..., 1]
print(consistency.shape)
fig, ax = subplots()
[ax.scatter(thetas, c) for c in consistency]
ax.scatter(thetas, tmp.mean(0))
ax.set_xlim(thetas[0], thetas[-1])
ax.set_xscale('log')
ax.legend(labels = 'MI_max Impact_max ='.split(), bbox_to_anchor = (1, 1.05))
ax.set_xscale('log');#  ax.set_yscale('log')
setp(ax, **dict(xlabel = r'$\theta$', ylabel = 'Consistency'))
