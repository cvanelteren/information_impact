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
#extractThis      = '1539187363.9923286' # this is 100
#extractThis      = '1540135977.9857328'
extractThis = extractThis if extractThis.startswith('/') else f"{dataPath}{extractThis}" 
data   = IO.extractData(extractThis)

# thetas = [10**-i for i in range(1, 20)]
thetas  = logspace(0, log10(finfo(float).eps), 100)
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
fitParam    = dict(maxfev = int(1e4), bounds = (0, inf))

mus = dd.mean(0)


aucs  = zeros((model.nNodes, len(thetas), 2))
roots = zeros((model.nNodes, len(thetas), 2))

MIN, MAX = mus.reshape(-1, 2).min(axis = 0), mus.reshape(-1, 2).max(axis =0)
mus = (mus - MIN) / (MAX - MIN)
x = arange(deltas // 2)
for idx in range(mus.shape[-1]):
    coeffs, errors = plotz.fit(mus[..., idx].T, func, params = fitParam)
    print(errors)
    for cidx, c in enumerate(coeffs):
        for tidx, theta in enumerate(thetas):
            m    = sum(c[:2])
            
#            theta =  theta * (m - c[0]) + c[0]     
#            theta = theta * (c[0] + c[1.]) 
#            tmpFunc = lambda x: abs(func(x, *c) - theta)
            
            rr =  ( log(c[1]) - log(theta - c[0]) ) / c[2]
            
#            root    = optimize.root(tmpFunc, 0, method = 'linearmixing')  
#            root = optimize.root(tmpFunc, rr, method = 'excitingmixing')
#            print(root, rr)
#            if root.success:
#                print('yes')
#            root = root.x 
            root = rr
                   
            root = 0 if root < 0 or isnan(root) else root
            xx   = linspace(0, root, 1000)
            auc =  integrate.simps(func(xx, *c), xx)
            
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
    coeffs, b = plotz.fit(mu, func)
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
[axi.set_ylabel(label, labelpad = 10) for axi, label in zip(ax.ravel(), labels)]
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
#zd = ndimage.filters.gaussian_filter1d(dd, .1, axis = 2)
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
rootsc     = zeros((dd.shape[0], model.nNodes, len(thetas), 2))
rootscData = zeros((dd.shape[0], model.nNodes, len(thetas), 2))

from scipy import interpolate
from time import sleep

x = arange(deltas // 2 + 1)

newx = linspace(0, deltas // 2 + 1, 100)

for idx, sample in enumerate(zd):
    for jdx, s in enumerate(sample.T):
        coeffs, errors = plotz.fit(s, func, params = fitParam)
        for kdx, c in enumerate(coeffs):
            m    = sum(c[:2])
            for zdx, theta in enumerate(thetas):
#                theta =  theta * (m - c[0])  # + c[0]
                rr = ( log(c[1]) - log(theta - c[0]) ) / c[2]
#                print(rr, theta, c)
                rr = 0 if rr < 0 or isnan(rr) else rr
                rootsc[idx, kdx, zdx, jdx] = rr
                
           
# %%
a = zd.mean(0)
fig, ax = subplots()
[ax.plot(i, color = c, label = model.rmapping[idx]) \
 for idx, (i, c) in enumerate(zip(a[...,0], colors))]
ax.set_yscale('log')
ax.set_title('MI')
ax.legend()

fig, ax = subplots()
[ax.plot(i, color = c, label = model.rmapping[idx]) \
 for idx, (i, c) in enumerate(zip(a[...,1], colors))]
ax.set_yscale('log')
ax.set_title('Hellinger')
ax.legend()
 # %% compute concistency
cons = lambda x : max(histogram(x, arange(-1, model.nNodes) + .5 , density = True)[0])
rootsc[isfinite(rootsc) == False] = nan

maxim = empty(tuple(i for idx, i in enumerate(rootsc.shape) if idx != 1))
maxim[:] = nan
for idx, r in enumerate(rootsc):
    for jdx, theta in enumerate(thetas):
            
        try:
            ma = nanargmax(r[:, jdx, :], axis = 0)
            maxim[idx, jdx, :] = ma
        except ValueError:
            continue
consistency = array([ [cons(i[isnan(i) != True]) for i in j] for j in maxim.T])

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

# %%
tmp = array([ \
     [\
      histogram(j[isnan(j) != True], bins = arange(-1, model.nNodes) + .5, density = True)[0]\
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
h = [axi.imshow(i.T, aspect = 'auto') for axi, i in zip(ax, maxim.T)]
[colorbar(hi, ax = axi) for hi, axi in zip(h, ax)]
# %% bar graph of the ranking
#fig, ax = subplots()
#tmp = zd[..., 0]
##tmp = ndimage.filters.gaussian_filter1d(tmp, .5, axis = -1)
#[[ax.plot(i, color = c, alpha = .4) for i, c in zip(j, colors)] for j in tmp]
#ax.set_yscale('log')


