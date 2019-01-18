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
from Utils import plotting as plotz, stats, IO
import os, re, networkx as nx, scipy

#close('all')
style.use('seaborn-poster')
dataPath = f"{os.getcwd()}/Data/"
#dataPath = '/mnt/'
extractThis      = IO.newest(dataPath)[-1]
#extractThis      = '1539187363.9923286'    # th.is is 100
#extractThis      = '1540135977.9857328'
extractThis = extractThis if extractThis.startswith('/') else f"{dataPath}{extractThis}"
data   = IO.DataLoader(extractThis)

# thetas = [10**-i for i in range(1, 20)]
thetas  = logspace(log10(.9), log10(finfo(float).eps), 100)
#thetas  = array([.5, .1, .01, .001])


temps    = list(data.keys())
temp     = temps[0]
pulseSize= list(data[temp].keys())[1]

print(f'Listing temps: {temps}')
print(f'Listing nudges: {pulseSize}')

figDir = '../thesis/figures/'
# %% Extract data

# Draw graph ; assumes the simulation is over 1 type of graph
from Models.fastIsing import Ising
control  = IO.loadData(data[temp]['control'][0]) # TEMP WORKAROUND
model    = Ising(control.graph)

# %%
fig, ax  = subplots(frameon = False)
positions = nx.nx_agraph.graphviz_layout(model.graph, prog = 'neato', \
                                         )
#                                         root = sorted(dict(model.graph.degree()).items())[0][0])

positions = {node : tuple(i * 1.2 for i in pos) for node, pos in positions.items() }
plotz.addGraphPretty(model, ax, positions, \
                     layout = dict(scale = 1),\
                     circle = dict(\
                                   radius = 15),\
                     annotate = dict(fontsize = 14),\
                     )
ax.axis('equal')
ax.set(xticks = [], yticks = [])
fig.show()
savefig(figDir + 'psychonetwork.eps', format = 'eps', dpi = 1000)

# %%
def f(**kwargs):
    for k, v in kwargs.items():
        print(k,v)
f(**dict(test = 1))
# %% 
settings = IO.readSettings(extractThis)
deltas   = settings['deltas']
repeats  = settings['repeat']


NTRIALS  = len(data[temp]['control'])
NODES    = model.nNodes
DELTAS   = deltas // 2 - 1 # throw away half for nudge
COND     = 2
THETAS   = thetas.size

# data matrix
#dd       = zeros((NTRIALS, NODES, DELTAS, COND))

# extract data for all nodes
information_impact = '$\mu_i$'
causal_impact      = '$\delta_i$'
from tqdm import tqdm
loadedData = {}
if 'results.pickle' in os.listdir():
    loadedData = IO.loadPickle('results.pickle')['loadedData']
    NTEMPS, NNUDGE, NTRIALS, NODES, DELTAS = loadedData.shape
else:
    NTEMPS = len(data)
    NNUDGE = len(data[temp])
    loadedData    = zeros((NTEMPS, NNUDGE, NTRIALS, NODES, DELTAS)) # change to loadedData
    for tidx, temp in enumerate(data):
        pulseCounter = 1 # skip control; work around
        for pulse in data[temp]:
            if pulse != 'control':
                for nudgedNode, fileNames in data[temp][pulse].items():
                    for idx, fileName in enumerate(tqdm(fileNames)):
                        
                        # load control; bias correct mi 
                        control = IO.loadData(data[temp]['control'][idx])
                        graph   = control.graph
                        # panzeri-treves correction
                        cpx = control.conditional
                        N   = repeats
                        mi  = control.mi
                        rs  = zeros(mi.shape)
                        for key, value in cpx.items():
                #                xx  = value[0] if isinstance(value, list) else value
                            for zdx, deltaInfo in enumerate(value):
                                for jdx, nodeInfo in enumerate(deltaInfo):
                                    rs[zdx, jdx] += plotz.pt_bayescount(nodeInfo, repeats) - 1
                #                rs += array([[plotz.pt_bayescount(k, repeats) - 1 for k in j]\
                #                              for j in value])
                        Rs = array([[plotz.pt_bayescount(j, repeats) - 1 for j in i]\
                                     for i in control.px])
                
                        bias = (rs - Rs) / (2 * repeats * log(2))
                        corrected = mi - bias
                        loadedData[tidx, 0, idx, ...] = corrected[:DELTAS, :].T
                        
                        # load nudge
                        sample  = IO.loadData(fileName)
                        px      = sample.px
                #        impact  = stats.hellingerDistance(control.px, px)
                        impact  = stats.KL(control.px, px)
                        # don't use +1 as the nudge has no effect at zero
                        redIm = nanmean(impact[DELTAS + 2:], axis = -1).T
                        # TODO: check if this works with tuples (not sure)
                        jdx = [model.mapping[int(j)] if j.isdigit() else model.mapping[j]\
                            for key in model.mapping\
                            for j in re.findall(str(key), re.sub(':(.*?)\}', '', nudgedNode))]
                        loadedData[tidx, pulseCounter, idx, jdx, :] = redIm.squeeze().T
                pulseCounter += 1
    IO.savePickle('results.pickle', dict(loadedData = loadedData,\
                                         data = data))
temps = [float(i.split('=')[-1]) for i in data.keys()]
nudges= list(data[next(iter(data))].keys())
# define color swaps
colors = cm.tab20(arange(NODES))
# swap default colors to one with more range
rcParams['axes.prop_cycle'] = cycler('color', colors)
# %% extract root from samples

# fit functions
#double = lambda x, a, b, c, d, e, f: a + b * exp(-c*(x)) + d * exp(- e * (x-f))
#double_= lambda x, b, c, d, e, f: b * exp(-c*(x)) + d * exp(- e * (x-f))
#single = lambda x, a, b, c : a + b * exp(-c * x)
#single_= lambda x, b, c : b * exp(-c * x)
#special= lambda x, a, b, c, d: a  + b * exp(- (x)**c - d)


import sympy as sy
from sympy import abc
from sympy.utilities.lambdify import lambdify
symbols     = (abc.x, abc.a, \
               abc.b, abc.c, \
               abc.d, abc.e,\
               abc.f)\
               
syF         = symbols[1] * sy.exp(- symbols[2] * symbols[0]) +\
 symbols[3] * sy.exp(- symbols[4] * (symbols[0] - symbols[5]))
print(syF)
#syF         = symbols[1] * sy.exp(- symbols[2] * symbols[0])
func        = lambdify(symbols, syF, 'numpy')

p0          = ones((func.__code__.co_argcount - 1)); # p0[0] = 0
fitParam    = dict(maxfev = int(1e6), bounds = (0, inf), p0 = p0)

settings = IO.readSettings(extractThis)
repeats  = settings['repeat']


# %% normalize data
from scipy import ndimage
zd = zeros(loadedData.shape)
for temp in range(NTEMPS):
    for nudge in range(NNUDGE):
        zdi = loadedData[temp, nudge] 
        zdi = ndimage.filters.gaussian_filter1d(zdi, 2, axis = -1)
#        zd = ndimage.filters.gaussian_filter1d(zd, 1, axis = -3)
        
        # scale data 0-1 along each sample (nodes x delta)
        rescale = True
#            rescale = False
        if rescale:
            zdi = zdi.reshape(zdi.shape[0], -1) # flatten over trials
            MIN, MAX = zdi.min(axis = 1), zdi.max(axis = 1)
            MIN = MIN[:, newaxis]
            MAX = MAX[:, newaxis]
            zdi = (zdi - MIN) / (MAX - MIN)
            zdi = zdi.reshape((NTRIALS, NODES, DELTAS))
            
        thresh = 1e-4
        #zd[zd <= thresh] = 0
        zdi[zdi <= finfo(zdi.dtype).eps] = 0 # remove everything below machine error
        # show means with spread
        zd[temp, nudge] = zdi
        
        


# plot means of all nudges and temperatures
# %%
fig, ax = subplots(3, 3, sharex = 'all', sharey = 'all')
mainax  = fig.add_subplot(111, frameon = 0)
mainax.set(xticks = [], yticks = [])
#mainax.set_title(f'\n\n').
mainax.set_xlabel('Time[step]', labelpad = 40)

x  = arange(DELTAS)
xx = linspace(0, 1 * DELTAS, 1000)
sidx = 2 # 1.96
labels = '$I(s_i^t ; S^t_0)$\tUnderwhelming\tOverwhelming'.split('\t')

from matplotlib.ticker import FormatStrFormatter
means, stds  = zd.mean(-3), zd.std(-3, ddof = 1)


for temp in range(NTEMPS):
    for nudge in range(NNUDGE):
        tax = ax[temp, nudge]
        if temp == 0:
            tax.set_title(labels[nudge])
        # compute mean fit
        mus    = means[temp, nudge]
        sigmas = stds[temp, nudge]
        meanCoeffs, meanErrors = plotz.fit(mus, func, params = fitParam)
        for node, idx in sorted(model.mapping.items(), key = lambda x : x[1]):
  
            # plot mean fit
            tax.plot(xx, func(xx, *meanCoeffs[idx]),\
                     color = colors[idx],\
                     alpha = .5, \
                     markeredgecolor = 'black',\
                     label =  node)
            # plot the raw data
            tax.errorbar(x, mus[idx],\
                         fmt = '.',\
                         yerr = sidx * sigmas[idx],\
                         markersize = 15, \
                         color = colors[idx],\
                         label = node) # mpl3 broke legends?
        
#            ax[cidx].set(xscale = 'log')

        # fill the standard deviation from mean
#        ax[cidx].fill_between(x, a, b, color = colors[idx], alpha  = .4)
#        ax[cidx].yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))

#    ax[cidx].set(yticks = (0, maxs_[..., cidx].max()))
#    ax[cidx].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 2))

    tax.text(.75, .9, \
             f'T={round(temps[temp], 2)}', \
             fontdict = dict(fontsize = 15),\
             transform = tax.transAxes,\
             horizontalalignment = 'right')
    tax.set(xlim = (-1.5, 30))
               
        #    ax[cidx].set(xscale = 'log', yscale = 'log'
# format plot
from mpl_toolkits.axes_grid1 import make_axes_locatable
mainax.legend(\
              tax.lines, [i.get_label() for i in tax.lines],\
              title          = 'Node', \
              title_fontsize = 15, \
              loc            = 'upper left', \
              bbox_to_anchor = (1, 1),\
              borderaxespad  = 0, \
              frameon        = False,\
              )

subplots_adjust(wspace = 0.01, hspace = 0.05, right = .8)
fig.show()
fig.savefig(figDir + 'mi_time.eps', format='eps', dpi=1000, pad_inches = 0,\
        bbox_inches = 'tight')
# %% estimate impact
#aucs       = zeros((NTRIALS, COND, NODES))

from scipy import interpolate
from time import sleep

COEFFS = zeros((COND, NTRIALS, NODES, p0.size))
x = arange(DELTAS)

lim = inf
#lim = DELTAS  // 2
#lim = np.inf

# uggly mp case

def worker(sample):
    coeffs, errors = plotz.fit(sample, func, params = fitParam)
    auc = zeros(len(sample))
#    print(coeffs.max(), coeffs.min())
    for nodei, c in enumerate(coeffs):
        tmp = syF.subs([(s, v) for s,v in zip(symbols[1:], c)])
        F   = lambda x: func(x, *c)
        tmp, _ = scipy.integrate.quad(F, 0, DELTAS)
#        tmp = sy.integrals.integrate(tmp, (abc.x, 0, DELTAS))
        auc[nodei] = tmp
    return auc
        
import multiprocessing as mp
with mp.Pool(mp.cpu_count()) as p:
    aucs = array(\
                p.map(\
                      worker, tqdm(zd.reshape(-1, NODES, DELTAS))\
                      )\
                 )
    aucs = aucs.reshape(tuple(i for i in loadedData.shape if i != DELTAS))

# %% show idt auc vs impact auc
fig, ax = subplots(3, 2, sharex = 'all', sharey = 'all')
subplots_adjust(hspace = 0, wspace = 0)
for temp in range(NTEMPS):
    for nudge in range(1, NNUDGE):
        tax = ax[temp, nudge - 1]
        for node, idx in model.mapping.items():
            tax.scatter(*aucs[temp, [0, nudge], :, idx], color = colors[idx])
#        tax.set(yscale = 'log').
#rcParams['axes.labelpad'] = 30
 # %% compute concistency

bins = arange(-.5, model.nNodes + .5)
cons = lambda x : nanmax(histogram(x, bins , density = True)[0])

ranking, maxim = stats.rankData(aucs)
consistency = array([ [cons(i) for i in j] for j in maxim.T])
# plot consistency of the estimator
naive = (maxim[...,0] == maxim[...,1])

# %% frequency heatmaps consistency
tmp = array([ \
     [\
      histogram(j, bins = bins, density = True)[0]\
      for j in i.T ]
     for i in maxim.T])
fig, ax = subplots(1, 2, sharey = 'all')
mainax  = fig.add_subplot(111, frameon = False,
                          xticks = [], yticks = [])
for axi, t in zip(ax, tmp):
    h = axi.imshow(t.T, aspect = 'auto', vmin = 0, vmax = 1)

colorbar(h, ax = axi, label = 'frequency')
mainax.set_xlabel(r'$\theta$', labelpad = 30)
# %%
def colorbar(mappable, **kwargs):
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, **kwargs)
fig, ax = subplots(1, COND, sharex = 'all', sharey = 'all')

mainax = fig.add_subplot(111, frameon = False)
mainax.set_title('Consistency of ranking over trials')
mainax.set(**dict(\
                  xticks = [],\
                  yticks = []))
mainax.set_xlabel('Trial number', labelpad = 10)
mainax.set_ylabel('Node idx', labelpad = 50)
subplots_adjust(wspace = .2)
for cidx in range(COND):
    h = ax[cidx].imshow(ranking[:, cidx, :].T)
    cbar = colorbar(h, ax = ax[cidx], anchor = (.5, 0))
cbar.set_label('Ranking')
# %%


from functools import partial
centralities = dict(degree      = nx.degree_centrality, \
             eigenvector = partial(nx.eigenvector_centrality, weight = 'weight'),\
             closeness   = nx.closeness_centrality,\
             betweenness = partial(nx.betweenness_centrality, weight = 'weight'),\
             )

fig, ax = subplots(2,2, sharex = 'all')
mainax = fig.add_subplot(111, frameon = False, xticks = [], yticks = [])
mainax.set_xlabel('Information impact', labelpad = 50)
mainax.set_ylabel('Centrality', labelpad = 50)
infImpact = aucs.mean(0)


# get the test statistic
ranking.sort()
target   = ranking[:, 1, -1]
rankings = ranking[:, 0, [-1]]

for i, (cent, cent_func) in enumerate(centralities .items()):
    print(idx)
    centrality = array(list(cent_func(model.graph).values()))
    rankings = hstack((rankings, argsort(centrality)[[-1]] * ones((len(ranking), 1))))
    tax = ax.ravel()[i]
    for idx, (impact, c) in enumerate(zip(infImpact, ['r', 'b'])):
        tax.scatter(impact, centrality)
#        tax.set(xscale = 'log')
    tax.set(title = cent)

tax.legend(['Information impact', 'Causal impact'], loc = 'upper left', \
  bbox_to_anchor = (1.01, 1), borderaxespad = 0)
subplots_adjust(hspace = .8)
#ax.set(xscale = 'log')

for label in get_figlabels():
    savefig(f'Figures/{label}')
show()

# %%
import scikit_posthocs as sp
percentage = percentage = array([i == target for i in rankings.T]).mean(1)

test = hstack((rankings, target[:, None]))
#test2 =
res =  scipy.stats.kruskal(*test.T)
print(res)
