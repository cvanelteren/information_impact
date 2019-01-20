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
pulseSize= list(data[temp].keys())

print(f'Listing temps: {temps}')
print(f'Listing nudges: {pulseSize}')

figDir = '../thesis/figures/'
# %% Extract data

# Draw graph ; assumes the simulation is over 1 type of graph
from Models.fastIsing import Ising
control  = IO.loadData(data[temp]['control'][0]) # TEMP WORKAROUND
model    = Ising(control.graph)
# %% # show mag vs temperature
tmp = IO.loadPickle(f'{extractThis}/mags.pickle')
fig, ax = subplots()
ax.scatter(tmp['temps'], tmp['mag'], alpha = .2)
ax.scatter(tmp['temperatures'], tmp['magRange'] * tmp['mag'].max(), \
           color = 'red', zorder = 2)

func = lambda x, a, b, c, d :  a / (1 + exp(b * (x - c))) + d # tanh(-a * x)* b + c
a, b = scipy.optimize.curve_fit(func, tmp['temps'], tmp['mag'].squeeze(), maxfev = 10000)
x = linspace(min(tmp['temps']), max(tmp['temps']), 1000)
ax.plot(x, func(x, *a), '--k')
ax.set(xlabel = 'Temperature (T)', ylabel = '|<M>|')
rcParams['axes.labelpad'] = 10
fig.savefig(figDir + 'temp_mag.eps', format = 'eps', dpi = 1000)
# %%
#fig, ax  = subplots(frameon = False)
#ax.set(xticks = [], yticks = [])
#positions = nx.nx_agraph.graphviz_layout(model.graph, prog = 'neato', \
#                                         )
##                                         root = sorted(dict(model.graph.degree()).items())[0][0])
#
#positions = {node : tuple(i * 1.2 for i in pos) for node, pos in positions.items() }
#plotz.addGraphPretty(model, ax, positions, \
#                     layout = dict(scale = 1),\
#                     circle = dict(\
#                                   radius = 15),\
#                     annotate = dict(fontsize = 14),\
#                     )
##ax.axis('equal')
#ax.set(xticks = [], yticks = [])
#fig.show()
#savefig(figDir + 'psychonetwork.eps', format = 'eps', dpi = 1000)

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
DELTAS   = deltas // 2 # throw away half for nudge
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
                        redIm = nanmean(impact[DELTAS:], axis = -1).T
                        # TODO: check if this works with tuples (not sure)
                        jdx = [model.mapping[int(j)] if j.isdigit() else model.mapping[j]\
                            for key in model.mapping\
                            for j in re.findall(str(key), re.sub(':(.*?)\}', '', nudgedNode))]
                        loadedData[tidx, pulseCounter, idx, jdx, :] = redIm.squeeze().T
                pulseCounter += 1
    if False:
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
double = lambda x, a, b, c, d, e, f: a + b * exp(-c*(x)) + d * exp(- e * (x-f))
double_= lambda x, b, c, d, e, f: b * exp(-c*(x)) + d * exp(- e * (x-f))
single = lambda x, a, b, c : a + b * exp(-c * x)
single_= lambda x, b, c : b * exp(-c * x)
special= lambda x, a, b, c, d: a  + b * exp(- (x)**c - d)


import sympy as sy
from sympy import abc
from sympy.utilities.lambdify import lambdify
symbols     = (abc.x, abc.a, \
               abc.b, abc.c, \
               abc.d, abc.e,\
               abc.f)\
               
#syF         = symbols[1] * sy.exp(- symbols[2] * symbols[0]) +\
# symbols[3] * sy.exp(- symbols[4] * (symbols[0] - symbols[5]))
#print(syF)
##syF         = symbols[1] * sy.exp(- symbols[2] * symbols[0])
#func        = lambdify(symbols, syF, 'numpy')
func        = double
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
#        zdi = ndimage.filters.gaussian_filter1d(zdi, 2, axis = -1)
        zd = ndimage.filters.gaussian_filter1d(zd, .2, axis = -3)
        
        # scale data 0-1 along each sample (nodes x delta)
#        rescale = True.
        rescale = False
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
        # remove the negative small numbers, e.g. -1e-5
        zd[temp, nudge] = abs(zdi) 
        
        

# %% estimate impact
#aucs       = zeros((NTRIALS, COND, NODES))

from scipy import interpolate
from time import sleep

COEFFS = zeros((COND, NTRIALS, NODES, p0.size))
x = arange(DELTAS)

# uggly mp case
def worker(sample):
    coeffs, errors = plotz.fit(sample, func, params = fitParam)
    auc = zeros((len(sample), 2))
    for nodei, c in enumerate(coeffs):
    #        tmp = syF.subs([(s, v) for s,v in zip(symbols[1:], c)])
    #        tmp = sy.integrals.integrate(tmp, (abc.x, 0, DELTAS))
        F   = lambda x: func(x, *c)
        tmp, _ = scipy.integrate.quad(F, 0, inf)

        auc[nodei, 0] = tmp
        auc[nodei, 1] = errors[nodei]
    return auc
        
import multiprocessing as mp
with mp.Pool(mp.cpu_count()) as p:
    aucs = array(\
                p.map(\
                      worker, tqdm(zd.reshape(-1, NODES, DELTAS))\
                      )\
                 )
                
rshape  = tuple(i for i in loadedData.shape if i != DELTAS)
rshape += (2,) # add error
tmp     = aucs.reshape(rshape)
aucs    = tmp[..., 0]
errors  = tmp[..., 1].mean(-2)


# %% remove outliers
stds_aucs = aucs.std(-2)
sidx = 1.5
aucs_corrected = zeros(aucs.shape)

tmp = aucs.mean(-2)
idx = where(abs(aucs - tmp[..., None, :]) > sidx * stds_aucs[..., None, :])
aucs_corrected = aucs.copy()
aucs_corrected[idx] = 0
# %% time plots
gs = dict(\
          height_ratios = [1, 1, 1], width_ratios = [1, .1, 1, 1])
fig, ax = subplots(3, 4, sharex = 'col', gridspec_kw = gs)
mainax  = fig.add_subplot(111, frameon = 0)
mainax.set(xticks = [], yticks = [])

#mainax.set_title(f'\n\n').
mainax.set_xlabel('Time[step]', labelpad = 40)

x  = arange(DELTAS)
xx = linspace(0, 1 * DELTAS, 1000)
sidx = 2 # 1.96
labels = ' \tUnderwhelming\tOverwhelming'.split('\t')
_ii    = '$I(s_i^t ; S^t_0)$'

[i.axis('off') for i in ax[:, 1]]
ax[1, 2].set_ylabel('KL-divergence', labelpad = 5)
ax[1, 0].set_ylabel(_ii, labelpad = 5)
from matplotlib.ticker import FormatStrFormatter
means, stds  = zd.mean(-3), zd.std(-3, ddof = 1)

for temp in range(NTEMPS):
    for nudge in range(NNUDGE):
        
        idx = nudge if nudge == 0 else nudge + 1
        tax = ax[temp, idx]
        if temp == 0:
            tax.set_title(labels[nudge] + '\n')
        # compute mean fit
        mus    = means[temp, nudge]
        sigmas = stds[temp, nudge]
        meanCoeffs, meanErrors = plotz.fit(mus, func, params = fitParam)
        for node, idx in sorted(model.mapping.items(), key = lambda x : x[1]):
  
            # plot mean fit
            tax.plot(xx, func(xx, *meanCoeffs[idx]),\
                     color = colors[idx],\
                     alpha = 1, \
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


        tax.ticklabel_format(axis = 'y', style = 'sci',\
                         scilimits = (0, 4))

    tax.text(.95, .9, \
             f'T={round(temps[temp], 2)}', \
             fontdict = dict(fontsize = 15),\
             transform = tax.transAxes,\
             horizontalalignment = 'right')
#    tax.set(xlim = (-1.5, 30))
               
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
subplots_adjust(hspace = .15, wspace = .2)
fig.show()
fig.savefig(figDir + 'mi_time.eps', format='eps', dpi=1000, pad_inches = 0,\
        bbox_inches = 'tight')


# %% show idt auc vs impact auc
fig, ax = subplots(3, 2, sharex = 'all')
subplots_adjust(hspace = .2, wspace = .2)
mainax = fig.add_subplot(111, frameon = False, \
                         xticks = [], yticks = [],\
                         xlabel = r'Information impact ($\mu_i$)',\
                         ylabel = r'Causal impact ($\delta_i$)'\
                         )
for temp in range(NTEMPS):
    for nudge in range(1, NNUDGE):
        tax = ax[temp, nudge - 1]
        tax.set(yscale = 'log')
        for node, idx in model.mapping.items():
            tax.scatter(*aucs[temp, [0, nudge], :, idx], \
                        color = colors[idx], \
                        label = node)
        if temp == 0:
            tax.set_title(labels[nudge])
            if nudge == NNUDGE - 1:
                tax.legend(loc = 'upper left', bbox_to_anchor = (1, 1),\
                           frameon = False, title = 'Node',
                           borderaxespad = 0,\
                           title_fontsize = 15)
        if nudge == NNUDGE - 1:
             tax.text(.95, .1, \
             f'T={round(temps[temp], 2)}', \
             fontdict = dict(fontsize = 15),\
             transform = tax.transAxes,\
             horizontalalignment = 'right')
#        tax.set(yscale = 'log')
            
        
#        tax.set(yscale = 'log').
rcParams['axes.labelpad'] = 80
savefig(figDir + 'causal_ii_scatter.eps', format = 'eps', dpi = 1000)
# %% appendix plot: fit error comparison
fig, ax = subplots()
errors = errors.reshape(-1, NODES)
subplots_adjust(hspace = 0)
width= .07
locs = arange(0, len(errors))



for node in range(NODES):
    ax.bar(locs + width * node, \
           errors[:, node], color = colors[node],\
           width = width, label = model.rmapping[node])
    
conditions = [f'T={round(x, 2)}\n{y}' for x in temps for y in nudges]  
# set labels per group and center it 
ax.set(yscale = 'log', xticklabels = conditions, \
       xticks = locs + .5 * NODES * width, ylabel = 'Mean square error')
ax.tick_params(axis = 'x', rotation = 45)
ax.legend(loc = 'upper left',\
          bbox_to_anchor = (1, 1),\
          borderaxespad = 0, \
          frameon = False)
rcParams['axes.labelpad'] = 1
fig.savefig(figDir + 'fiterror.eps', format = 'eps', dpi = 1000)

# %%
def colorbar(mappable, **kwargs):
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, **kwargs)

# %% information impact vs centrality
from functools import partial
centralities = dict(deg      = partial(model.graph.degree(weight = 'weight')), \
             close   = nx.closeness_centrality,\
             bet = partial(nx.betweenness_centrality, weight = 'weight'),\
              ev = partial(nx.eigenvector_centrality, weight = 'weight'),\
             )


infImpact = aucs.mean(-2)

# get the test statistic
from mpl_toolkits.mplot3d import Axes3D
ranking.sort()

conditionLabels = 'Underwhelming Overwhelming'.split()

    
# plot info
for condition, condLabel in enumerate(conditionLabels):
    fig, ax = subplots(4, 3, sharex = 'all', sharey = 'row')
    mainax = fig.add_subplot(111, frameon = False, xticks = [], yticks = [])
    mainax.set_xlabel('Causal impact', labelpad = 50)
    for i, (cent, cent_func) in enumerate(centralities .items()):
        # compute cents
        centrality = array(list(dict(cent_func(model.graph)).values()))
#        rankCentrality[i] = argsort(centrality)[-1]
#        rankCentrality = zeros((len(centralities)))
        for temp in range(NTEMPS):
            tax = ax[i, temp]
#            tax.set(yscale = 'log')
            # scale node according to information impact
            tmp = infImpact[temp, 0]
            tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            for node in range(NODES):
                tax.scatter(\
                            infImpact[temp, condition + 1, node], \
                            centrality[node], \
                            color = colors[node], \
                            label = model.rmapping[node],\
                            s = 300  * tmp[node])
            # every column set title
            if temp == 0:
                tax.text(-.35, .5, cent, fontdict = dict(\
                                                  fontsize = 15),\
                     transform = tax.transAxes,\
                     horizontalalignment = 'center')
            # every row set label
            if i == 0:
                tax.set_title(f'T={round(temps[temp], 2)}')
                if temp == NTEMPS - 1:
                    leg = tax.legend(loc = 'upper left', \
                              bbox_to_anchor = (1.0, 1),\
                              borderaxespad = 0, frameon = False,\
                              title = 'Node',\
                              title_fontsize = 15,)

 
    # hack for uniform labels
    for l in leg.legendHandles:
        l._sizes = [150]
    mainax.set_title(condLabel + '\n\n')
    fig.canvas.draw()
    #tax.legend(['Information impact', 'Causal impact'], loc = 'upper left', \
    #  bbox_to_anchor = (1.01, 1), borderaxespad = 0)
    subplots_adjust(wspace = .1, hspace = .1)
    rcParams['axes.labelpad'] = 20
    #ax.set(xscale = 'log')
    fig.savefig(figDir + f'causal_cent_ii_{conditionLabels[condition]}.eps', \
               format = 'eps', dpi = 1000)



# %%

drivers = argmax(aucs, -1)

percentage = zeros((NTEMPS, COND))
percstds   = zeros((NTEMPS, COND))
for cond in range(1, COND):
    percentage[:, cond - 1] = (drivers[:, 0] == drivers[:, cond]).mean(1)
    percstds[:, cond - 1] = (drivers[:, 0] == drivers[:, cond]).std(1)
    centApprox = zeros((NTEMPS, len(centralities), COND))
    for cidx, (cent, centF) in enumerate(centralities.items()):
        centMax  = argmax(array(list(dict(centF(model.graph)).values())))
        print(centMax)
        centMax *= ones(NTRIALS)
        centApprox[:, cidx, cond - 1] = (drivers[:, 1] == centMax).mean(1)
# group data by temp or by temp?
locs     = arange(NNUDGE - 1) * 2
width    = .4
tempLocs = linspace(0, 1 + len(centralities) * width, NTEMPS)

fig, ax = subplots(1, 2,  sharey = 'all')
mainax = fig.add_subplot(111, frameon = False, xticks = [], yticks = [])
mainax.set(ylabel ='prediction accuracy(%)' )
for cond in range(NNUDGE - 1):  
    tax = ax[cond]
    tax.bar(tempLocs, percentage[:, cond] * 100, width = width,\
               color = colors[0], label = 'max $\mu_i$')
    for cidx, cent in enumerate(centApprox[..., cond].T):
        tax.bar(tempLocs + width * (cidx + 1), cent * 100, width = width, \
               color = colors[cidx + 1], label = list(centralities.keys())[cidx])
    tax.set_title(conditionLabels[cond])   
    tax.set(xticks = tempLocs + .5 * 4 * width, \
            xticklabels = [f'T={round(i,2)}' for i in temps])
    tax.tick_params(axis = 'x', rotation = 45)
    
tax.legend(loc = 'upper left', \
                              bbox_to_anchor = (1.0, 1),\
                              borderaxespad = 0, frameon = False,\
                              title = 'Method',\
                              title_fontsize = 15,) 
#tax.get_yaxis().set_visible(False)     


subplots_adjust(wspace = 0)
rcParams['axes.labelpad'] = 60
#ax.set(yscale = 'symlog')
#ax.set(xticklabels = conditionLabels)

# %%

import scikit_posthocs as sp
# %%
import scikit_posthocs as sp
percentage = percentage = array([i == target for i in rankings.T]).mean(1)

test = hstack((rankings, target[:, None]))
#test2 =
res =  scipy.stats.kruskal(*test.T)
print(res)

 # %% compute concistency

bins = arange(-.5, model.nNodes + .5)
cons = lambda x : nanmax(histogram(x, bins , density = True)[0])

ranking, maxim = stats.rankData(aucs)
consistency = array([ [cons(i) for i in j] for j in maxim.T])
# plot consistency of the estimator
naive = (maxim[...,0] == maxim[...,1])
