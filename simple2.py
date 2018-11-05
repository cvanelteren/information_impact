#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:09:36 2018

@author: casper
"""
from numpy import *
from scipy import optimize, integrate
from matplotlib.pyplot import *
from time import sleep

import IO, os, plotting as plotz, stats

style.use('seaborn-poster')
dataPath = f"{os.getcwd()}/Data/"
extractThis      = IO.newest(dataPath)[-1]
#extractThis      = '1539187363.9923286' # this is 100
#extractThis      = '1540135977.9857328'
extractThis = extractThis if extractThis.startswith('/') else f"{dataPath}{extractThis}" 
data   = IO.extractData(extractThis)

# thetas = [10**-i for i in range(1, 20)]
thetas  = logspace(0, log10(finfo(float).eps), 100)
#thetas  = array([.5, .1, .01, .001])
t       = next(iter(data))
model   = data[t]['{}'][0].model

# %% Extract data
deltas   = IO.readSettings(extractThis)['deltas']
controls = array([i.mi for i in data[t]['{}']])
roots    = zeros((len(controls), model.nNodes, len(thetas), 2))

colors = cm.tab20(arange(model.nNodes))

NSAMPLES, NODES, DELTAS, COND = len(controls), model.nNodes, deltas // 2 + 1, 2
dd = zeros((NSAMPLES, NODES, DELTAS, COND))
for condition, samples in data[t].items():
    for idx, sample in enumerate(samples):
        if condition == '{}':
            dd[idx, ..., 0] = sample.mi[:deltas // 2 + 1, :].T
#            ax.plot(sample.mi)
        else:
            control = data[t]['{}'][idx].px
            impact = stats.hellingerDistance(sample.px, control).mean(-1)
#            impact = stats.KL(control, sample.px).mean(-1)
            impact = impact[deltas // 2  : ][None, :].T
            # TODO: check if this works with tuples (not sure)
            jdx = [model.mapping[int(j)] if j.isdigit() else model.mapping[j]\
                for key in model.mapping\
                for j in re.findall(str(key), re.sub(':(.*?)\}', '', condition))]
            dd[idx, jdx, ...,  1] = impact.squeeze().T
#ax.set_xlim(0, 4)        
# %% extract root from samples
            
# fit functions
double = lambda x, a, b, c, d, e, f: a + b * exp(-c*x) + d * exp(- e * (x-f))
single = lambda x, a, b, c : a + b * exp(-c * x)
single_= lambda x, b, c : b * exp(-c * x)
func   = single
p0          = ones((func.__code__.co_argcount - 1)); # p0[0] = 0
fitParam    = dict(maxfev = int(1e4), bounds = (0, inf))

mus = dd.mean(0)



roots = zeros((model.nNodes, len(thetas), dd.shape[-1] + 2))
aucs  = zeros(roots.shape)
MIN, MAX = mus.reshape(-1, 2).min(axis = 0), mus.reshape(-1, 2).max(axis =0)
mus = (mus - MIN) / (MAX - MIN)
x = arange(deltas // 2)
for idx in range(mus.shape[-1]):
    coeffs, errors = plotz.fit(mus[..., idx].T, func, params = fitParam)
    for cidx, c in enumerate(coeffs):
        m = sum(c[:2])
        for tidx, theta in enumerate(thetas):
            theta =  theta * (m - c[0]) + c[0]
            root =  ( log(c[1]) - log(theta - c[0]) ) / c[2]
            root = 0 if root < 0 or isnan(root) else root
            xx   = linspace(0, root, 1000)
            auc, er =  integrate.quad(lambda x : func(x, *c), 0, root)
            
            roots[cidx, tidx, idx] = root
            aucs[cidx, tidx, idx]  = 0 if isnan(auc) else auc
# %% time plots
fig, axes = subplots(2, sharex = 'all')
mainAx  = fig.add_subplot(111, frameon = False)
setp(mainAx, **dict(xticks = [], yticks = []))
mainAx.set_xlabel('time [step]', labelpad = 20)
newx = linspace(0, deltas // 2 + 1, 100)
from scipy import ndimage
for ax, mu in zip(axes.ravel(), mus.T):
    [ax.plot(mui, color = c) for mui, c in zip(mu.T, colors)]
    coeffs, b = plotz.fit(mu, func, params = fitParam)
    [ax.plot(newx, func(newx, *c), '--', color = col) for c, col in \
     zip(coeffs, colors)]
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
    [axi.plot(thetas, ii, color = c) \
     for idx, (ii, c) in enumerate(zip(i.T, colors))]
    axi.set_yscale('log')
    axi.set_xscale('log')
    axi.set_xlim(thetas[0], thetas[-1])
labels = r'$AUC_{MI}$ $AUC_{<H>}$'.split()

[axi.set_ylabel(label, labelpad = 10) \
 for axi, label in zip(ax.ravel(), labels)]

mainAx.set_xlabel(r'$\theta$', labelpad = 20)
mainAx.legend(handles = axi.lines, labels = [node for node in model.graph.nodes()], \
                        bbox_to_anchor = (1.1, 1), title = 'Node')
mainAx.axis('off')
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
#from mpl_toolkits.mplot3d import Axes3D
#fig, ax = subplots(subplot_kw = dict(projection = '3d'))
#for c, x in zip(colors, roots):
#    ax.scatter(*log10(x.T), zs = log10(thetas), color = c, alpha =1)
    
# %% concistency
zd = dd;
#zd = ndimage.filters.gaussian_filter1d(zd, 2, axis = -2)
zd[zd < finfo(float).eps] = 0
zd = zd.reshape(zd.shape[0], -1, zd.shape[-1])

MIN, MAX = zd.min(axis = 1), zd.max(axis = 1)
MIN = MIN[:, newaxis, :]
MAX = MAX[:, newaxis, :]
zd = (zd - MIN) / (MAX - MIN) 
zd = zd.reshape(dd.shape)

# %% root based on data
p0             = ones((func.__code__.co_argcount - 1)); p0[0] = 0
fitParam['p0'] = p0
rootsc      = zeros((dd.shape[0], model.nNodes, len(thetas), dd.shape[-1]))
aucs        = zeros(rootsc.shape)
from scipy import interpolate
from time import sleep

COEFFS = zeros((COND, NSAMPLES, NODES, p0.size))
x = arange(deltas // 2 + 1)

newx = linspace(0, deltas // 2 + 1, 100)

for idx, sample in enumerate(zd):
    for jdx, s in enumerate(sample.T):
        coeffs, errors = plotz.fit(s, func, params = fitParam)
        COEFFS[jdx, idx, ...] = coeffs
        for kdx, c in enumerate(coeffs):
            m    = sum(c[:2])
            for zdx, theta in enumerate(thetas):
                theta =  theta * (m - c[0]) + c[0]
#                rr optimize.fsolve(func, 0)
                root = ( log(c[1]) - log(theta - c[0]) ) / c[2]
                root = 0 if root < 0 or isnan(root) else root
                
                auc, er =  integrate.quad(lambda x : func(x, *c), 0, root)
                auc = 0 if auc < 0 or isnan(auc) else auc
                rootsc[idx, kdx, zdx, jdx] = root
                aucs[idx, kdx, zdx, jdx]   = auc
                
 # %% compute concistency
bins = arange(-.5, model.nNodes + .5)
cons = lambda x : max(histogram(x, bins , density = True)[0])

maxim = empty(tuple(i for idx, i in enumerate(rootsc.shape) if idx != 1))
maxim[:] = nan
for idx, r in enumerate(aucs):
    for jdx, theta in enumerate(thetas):
            
        try:
            ma = nanargmax(r[:, jdx, :], axis = 0)
            maxim[idx, jdx, :] = ma
        except ValueError:
            print('found no max')
            continue
consistency = array([ [cons(i[isnan(i) != True]) for i in j] for j in maxim.T])
print(unique(maxim))
# plot consistency of the estimator
naive = (maxim[...,0] == maxim[...,1]).mean(0)
fig, ax = subplots()
[ax.plot(thetas, c, alpha = .5) for c in consistency]
ax.plot(thetas, naive)
ax.set_xlim(thetas[0], thetas[-1])
ax.legend(labels = 'MI_max Impact_max ='.split(), bbox_to_anchor = (1.25, 1.0))
ax.set_xscale('log'); # ax.set_yscale('log')
setp(ax, **dict(xlabel = r'$\theta$', ylabel = 'Consistency', \
                title = 'Max based on regression'))

# %% frequency heatmaps
tmp = array([ \
     [\
      histogram(j[isnan(j) != True], bins = bins, density = True)[0]\
      for j in i ]
     for i in maxim.T])
fig, ax = subplots()
h = ax.imshow(tmp[0].T, aspect = 'auto')
ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'node label')
ax.set_title('MI')
colorbar(h, ax = ax, label = 'Frequency')

fig, ax = subplots()
h = ax.imshow(tmp[1].T, aspect = 'auto')
ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'node label')
ax.set_title('Hellinger')
colorbar(h, ax = ax, label = 'Frequency')

# %%
fig, ax = subplots(2)
for axi, d in zip(ax, maxim.T):
    h = axi.imshow(d.T, aspect = 'auto')
#    axi.set_aspect('equal')
    s = d.T.shape
#    axi.set_xticks(arange(s[1]))
#    axi.set_yticks(arange(s[0]))
#    axi.set_xticklabels([])
#    axi.set_yticklabels([])
#    
#    axi.set_xticks(arange(-.5, s[1]), minor = 1)
#    axi.set_yticks(arange(-.5, s[0]), minor = 1)
    colorbar(h, ax = axi)
    axi.grid(which = 'minor', color = 'k', linestyle = '-', linewidth = 2)

# %% bar graph of the ranking
fig, ax = subplots(2)
x = linspace(0, deltas // 2 + 1, 1000)

# for all conditions
for axi, tmp in zip(ax, COEFFS):
    # for all samples
    for tmpi in tmp:
        [axi.plot(x, func(x, *c), color = col, alpha = .3) for col, c in zip(colors, tmpi)]
    axi.set_yscale('log')
#ax.set_yscale('log')
# %%
fig, ax = subplots()
tmp = mean(aucs, axis = 0)
[ax.scatter(i[...,0], i[...,1], color = c, alpha = .5)for c, i in zip(colors, tmp)]
ax.set_yscale('log')
# %%
fig, ax = subplots()
for s in aucs:
    for color, node in zip(colors, s):
        ax.scatter(node[...,0], node[...,1], color = color, alpha = .5)
        
# %% 
from mpl_toolkits.mplot3d import Axes3D 
fig, ax = subplots(subplot_kw = dict(projection = '3d'))
for s in aucs:
    for nodeData, c, in zip(s, colors):
        ax.scatter(*nodeData.T, thetas, color = c)
setp(ax, **dict(xlabel = 'mi auc', ylabel = 'impact auc', zlabel = 'theta'))
    
