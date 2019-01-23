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
import os, re, networkx as nx, scipy, multiprocessing as mp

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

settings = IO.readSettings(extractThis)
deltas   = settings['deltas']
repeats  = settings['repeat']


temps    = [float(i.split('=')[-1]) for i in data.keys()]
nudges   = list(data[next(iter(data))].keys())
temp     = temps[0]

# Draw graph ; assumes the simulation is over 1 type of graph
from Models.fastIsing import Ising
control  = IO.loadData(data[f't={temp}']['control'][0]) # TEMP WORKAROUND
model    = Ising(control.graph)

NTRIALS  = len(data[f't={temp}']['control'])
NTEMPS   = len(data)
NNUDGE   = len(data[f't={temp}'])
NODES    = model.nNodes
DELTAS   = deltas // 2 # throw away half for nudge
COND     = 2
THETAS   = thetas.size

pulseSizes = list(data[f't={temp}'].keys())

print(f'Listing temps: {temps}')
print(f'Listing nudges: {pulseSizes}')

figDir = '../thesis/figures/'
# %% Extract data

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


# data matrix
#dd       = zeros((NTRIALS, NODES, DELTAS, COND))

# extract data for all nodes
information_impact = '$\mu_i$'
causal_impact      = '$\delta_i$'
from tqdm import tqdm


# define color swaps
colors = cm.tab20(arange(NODES))
# swap default colors to one with more range
rcParams['axes.prop_cycle'] = cycler('color', colors)

# %% Need an alternative for this badly...
def flattenDict(d):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += flattenDict(v)
        else:
            out.append(v)
    return out

def worker(fidx):
    fileName = fileNames[fidx]
    # do control
    data = frombuffer(var_dict.get('X')).reshape(var_dict['xShape'])
    
    node, temp, trial = unravel_index(fidx, var_dict.get('start'), order = 'F')
    
    if '{}' in fileName:
        # load control; bias correct mi 
        control = IO.loadData(fileName)
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
        data[0, trial, temp]  = corrected[:DELTAS, :].T
    else:
        # load control
        jdx = [model.mapping[int(j)] if j.isdigit() else model.mapping[j]\
                             for key in model.mapping\
                             for j in re.findall(str(key), re.sub(':(.*?)\}', '', fileName))]
        jdx = jdx[0]
        useThis = fidx - node
        control = IO.loadData(fileNames[useThis])
        # load nudge
        sample  = IO.loadData(fileName)
        px      = sample.px
        impact  = stats.hellingerDistance(control.px, px)
#        impact  = stats.KL(control.px, px)
        # don't use +1 as the nudge has no effect at zero
        redIm = nanmean(impact[DELTAS:], axis = -1).T
        # TODO: check if this works with tuples (not sure)
        data[(node - 1) // NODES + 1, trial, temp, jdx,  :] = redIm.squeeze().T

# look for stored data [faster]
try:
    for k, v in IO.loadPickle(f'{extractThis}/results.pickle'):
        globals()[k] = v 
        
    NNUDGE, NTEMPS, NTRIALS, NODES, DELTAS = loadedData.shape
    
# start loading individual pickles
except:
    fileNames = sorted(\
                       [j for i in flattenDict(data) for j in i],\
                       key = lambda x: os.path.getctime(x),\
                       )
    var_dict = {}
    def initWorker(X, xShape, start):
        var_dict['X']      = X
        var_dict['xShape'] = xShape
        var_dict['start']  = start
    expStruct = (NODES * COND  + 1, NTRIALS, NNUDGE)
    buffShape = (NNUDGE, NTEMPS, NTRIALS, NODES, DELTAS)
    buff = mp.RawArray('d', int(prod(buffShape)))

    tmp = range(len(fileNames))
    with mp.Pool(processes = mp.cpu_count() - 1, initializer = initWorker,\
             initargs = (buff, buffShape, expStruct)) as p:
        p.map(worker, tqdm(tmp))
    loadedData = frombuffer(buff, dtype = float64).reshape(*buffShape)
    # store for later
    IO.savePickle(f'{extractThis}/results.pickle', dict(\
                  loadedData = loadedData, data = data))
    del buff
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
fitParam    = dict(maxfev = int(1e6), \
                   bounds = (0, inf), p0 = p0,\
                   jac = 'cs')

settings = IO.readSettings(extractThis)
repeats  = settings['repeat']


# %% normalize data

from scipy import ndimage
zd = zeros(loadedData.shape)
for temp in range(NTEMPS):
    for nudge in range(NNUDGE):
        zdi = loadedData[nudge, temp] 
        # scale data 0-1 along each sample (nodes x delta)
        rescale = True
#        rescale = False
        # rescale for each trial over min / max
#        zdi = ndimage.filters.gaussian_filter1d(zdi, 1, axis = -1)
#        zdi = ndimage.filters.gaussian_filter1d(zdi, 3, axis = 0)
        if rescale:
            zdi = zdi.reshape(NTRIALS, -1) # flatten over trials
            MIN, MAX = zdi.min(axis = 1), zdi.max(axis = 1)
            MIN = MIN[:, newaxis]
            MAX = MAX[:, newaxis]
            zdi = (zdi - MIN) / (MAX - MIN)
            zdi = zdi.reshape((NTRIALS, NODES, DELTAS))

        thresh = 1e-4
        #zd[zd <= thresh] = 0
        zdi[zdi <= finfo(zdi.dtype).eps] = 0 # remove everything below machine error
#        zdi[zdi < 0]  = 0
        # show means with spread
        # remove the negative small numbers, e.g. -1e-5
        zd[temp, nudge] = zdi 
        
        

# %% estimate impact
#aucs       = zeros((NTRIALS, COND, NODES))

from scipy import interpolate
from time import sleep

COEFFS = zeros((COND, NTRIALS, NODES, p0.size))
x = arange(DELTAS)

# uggly mp case
def worker(sample):
    auc = zeros((len(sample), 2))
    coeffs, errors = plotz.fit(sample, func, params = fitParam)
    for nodei, c in enumerate(coeffs):
    #        tmp = syF.subs([(s, v) for s,v in zip(symbols[1:], c)])
    #        tmp = sy.integrals.integrate(tmp, (abc.x, 0, DELTAS))
        F   = lambda x: func(x, *c) - c[0]
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

# remove machine error estimates
aucs[aucs <= finfo(aucs.dtype).eps ] = 0


# %% remove outliers
stds_aucs = aucs.std(-2)
sidx = 1.5
aucs_corrected = zeros(aucs.shape)

tmp = aucs.mean(-2)
idx = where(abs(aucs - tmp[..., None, :]) > sidx * stds_aucs[..., None, :])
aucs_corrected = aucs.copy()
aucs_corrected[idx] = 0
# %% time plots

# insert dummy axis to offset subplots
gs = dict(\
          height_ratios = [1, 1, 1], width_ratios = [1, .15, 1, 1],\
          )
fig, ax = subplots(3, 4, sharex = 'col', sharey = 'all', gridspec_kw = gs)
mainax  = fig.add_subplot(111, frameon = 0)
mainax.set(xticks = [], yticks = [])

#mainax.set_title(f'\n\n').
mainax.set_xlabel('Time[step]', labelpad = 40)

x  = arange(DELTAS)
xx = linspace(0, 1 * DELTAS, 1000)
sidx = 2 # 1.96
labels = 'Control\tUnderwhelming\tOverwhelming'.split('\t')
_ii    = '$I(s_i^t ; S^t_0)$'

[i.axis('off') for i in ax[:, 1]]

rcParams['axes.labelpad'] = 80
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
subplots_adjust(hspace = 0, wspace = 0)
fig.show()
fig.savefig(figDir + 'mi_time.eps', format='eps', dpi=1000, pad_inches = 0,\
        bbox_inches = 'tight')


# %% show idt auc vs impact auc

from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor as lof
clf = EllipticEnvelope()
rcParams['axes.labelpad'] = 40
fig, ax = subplots(3, 2, sharex = 'all', sharey = 'row' )
subplots_adjust(hspace = 0, wspace = 0)
mainax = fig.add_subplot(111, frameon = False, \
                         xticks = [], yticks = [],\
                         xlabel = r'Information impact ($\mu_i$)',\
                         ylabel = r'Causal impact ($\delta_i$)'\
                         )
out = lof(n_neighbors = NTRIALS)
for temp in range(NTEMPS):
    for nudge in range(1, NNUDGE):
        tax = ax[temp, nudge - 1]
#        tax.set(yscale = 'log')
#        scope_max = aucs[temp, [0, nudge]].ravel().max(0)
#        scope_min = aucs[temp, [0, nudge]].ravel().min(0)
#        ranges = linspace(scope_min, scope_max, 100)
#        
#        xx, yy = meshgrid(ranges, ranges)
        for node, idx in model.mapping.items():
            tmp = aucs[temp, [0, nudge], :, idx]
#            clf.fit(tmp.T)
#            out.fit(tmp.T)
#            rr = out.negative_outlier_factor_
            
#            radius = (rr.max() - rr) / (rr.max() - rr.min())
#            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#            tax.scatter(*tmp, s = radius * 100, \
#                        edgecolor = colors[idx], facecolor = 'none')
#            tax.contour(xx, yy, Z.reshape(xx.shape), \
#                        colors = colors[[idx]],\
#                        edgecolor = 'k',\
#                        levels = [0], linewidths = 2)
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
             tax.text(.95, .9, \
             f'T={round(temps[temp], 2)}', \
             fontdict = dict(fontsize = 15),\
             transform = tax.transAxes,\
             horizontalalignment = 'right')
#        tax.set(yscale = 'log')
            
        
#        tax.set(yscale = 'log').

savefig(figDir + 'causal_ii_scatter.eps', format = 'eps', dpi = 1000)

# %% information impact vs centrality
from functools import partial
centralities = dict(deg      = partial(nx.degree, weight = 'weight'), \
             close   = nx.closeness_centrality,\
             bet = partial(nx.betweenness_centrality, weight = 'weight'),\
              ev = partial(nx.eigenvector_centrality, weight = 'weight'),\
             )


infImpact = aucs.mean(-2)

# get the test statistic
from mpl_toolkits.mplot3d import Axes3D
#ranking.sort()

conditionLabels = 'Underwhelming Overwhelming'.split()

    
# plot info
rcParams['axes.labelpad'] = 20
for condition, condLabel in enumerate(conditionLabels):
    fig, ax = subplots(4, 3, sharex = 'all', sharey = 'row')
    mainax = fig.add_subplot(111, frameon = False, xticks = [], yticks = [])
    mainax.set_xlabel('Causal impact', labelpad = 50)
    for i, (cent, cent_func) in enumerate(centralities .items()):
        # compute cents
        centrality = dict(cent_func(model.graph))
#        rankCentrality[i] = argsort(centrality)[-1]
#        rankCentrality = zeros((len(centralities)))
        for temp in range(NTEMPS):
            tax = ax[i, temp]
#            tax.set(xscale = 'log')
            # scale node according to information impact
            tmp = infImpact[temp, 0]
            tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            for node in range(NODES):
                nl = model.rmapping[node]
                tax.scatter(\
                            infImpact[temp, condition + 1, node], \
                            centrality[nl], \
                            color = colors[node], \
                            label = model.rmapping[node],\
                            s = max((600  * tmp[node], 10)), alpha = 1)
            # every column set title
            if temp == 0:
                tax.text(-.35, .5, cent, fontdict = dict(\
                                                  fontsize = 17),\
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
                              title_fontsize = 15, fontsize = 12)

 
    # hack for uniform labels
    for l in leg.legendHandles:
        l._sizes = [150]
    mainax.set_title(condLabel + '\n\n')
    fig.canvas.draw()
    #tax.legend(['Information impact', 'Causal impact'], loc = 'upper left', \
    #  bbox_to_anchor = (1.01, 1), borderaxespad = 0)
    subplots_adjust(wspace = .1, hspace = .1)
    #ax.set(xscale = 'log')
    fig.savefig(figDir + f'causal_cent_ii_{conditionLabels[condition]}.eps', \
               format = 'eps', dpi = 1000)



# %%

drivers = argmax(aucs, -1)
all_drivers = drivers.reshape(-1, NTRIALS)
N = len(centralities)
centApprox = zeros((NTEMPS, N, COND))


percentages = zeros((NTEMPS, N + 1, COND))

estimates = zeros((NTEMPS, NTRIALS,  N + 1, NODES))
maxestimates = zeros((NTEMPS, NTRIALS, N + 1), dtype = int)
targets   = zeros((NTEMPS, NTRIALS, COND, NODES))
for temp in range(NTEMPS):
    for cond in range(COND):
        percentages[temp, 0, cond] = equal(drivers[temp, 0], drivers[temp, cond + 1]).mean()
        estimates[temp, :, 0, :]   = aucs[temp, 0]
        targets[temp, :, cond]     = aucs[temp, cond + 1]
        maxestimates[temp, :, 0 ]  = aucs[temp, 0].argmax(-1)
        for ni, centF in enumerate(centralities.values()):
            tmp = dict(centF(model.graph))
            centLabel, centEstimate = sorted(tmp.items(), key = lambda x: x[1])[-1]
            
            percentages[temp, ni + 1, cond] =  (centEstimate * ones(NTRIALS) == drivers[temp, cond + 1]).mean()
            
            maxestimates[temp, :, ni + 1] = model.mapping[centLabel] * ones(NTRIALS)
            for k, v in tmp.items():
                estimates[temp, :, ni + 1, model.mapping[k]] = v * ones((NTRIALS))
# plot max only 
    
fig, ax = subplots(2, sharex = 'all')

m = estimates.max(-1)
n = targets.max(-1)
rcParams['axes.labelpad'] = 5
for temp in range(NTEMPS):
    for ni in range(2):
        ax[0].scatter(n[temp, :, 0],  m[temp, :, ni])
        ax[1].scatter(n[temp, :, 1], m[temp, :, ni])
xlabel('causal')

# %%
tmp = zeros((NTEMPS, N + 1, COND))
fig, ax = subplots(1, 2, sharex = 'all', sharey = 'all')
rcParams['axes.labelpad'] = 50
mainax = fig.add_subplot(111, frameon = False, \
                         xticks = [],\
                         yticks = [],\
                         ylabel = 'Predicition accuracy')
#mainax.set(ylabel = "Prediction accuracy")
subplots_adjust(wspace = 0)
width = .4
xloc  = arange(NTEMPS) * 2
conditions = [f'T={round(x, 2)}\n{y}' for x in temps for y in nudges]  
condLabels = 'Underwhelming Overwhelming'.split()
labels = 'max $\mu_i$\tdeg\tclose\tbet\tev'.split('\t')

maxT = argmax(targets,  -1)
for cond in range(COND):
    tax = ax[cond]
    tax.set_title(condLabels[cond])
    tax.set(xticks = xloc + .5 * width * N, \
            xticklabels = conditions,\
            )
    tax.tick_params(axis = 'x', rotation = 45)
    
    for temp in range(NTEMPS):
        for ni in range(N + 1):
        
            x = equal(maxestimates[temp, :, ni], maxT[temp, :, cond])
            y = x.mean() * 100
            x = xloc[temp]
            
            tmp[temp, ni, cond] =  x.mean()
            tax.bar(x + width * ni, y, width = width,\
              color = colors[ni])  
            if y > 0:
                tax.text(x + width * ni, y + .5, \
                     int(y),\
                     fontweight = 'bold', \
                     horizontalalignment = 'center')

           
# %% cross validation
from sklearn.model_selection import LeaveOneOut
from sklearn import svm, metrics, linear_model as lm

#clf = svm.SVC(decision_function_shape = 'ovr', gamma = 'scale')
#clf = lm.LogisticRegression(\
#                            solver = 'lbfgs',\
#                            multi_class = 'multinomial')
clf = lm.LogisticRegression(C = 1, solver = 'lbfgs')
from sklearn.model_selection import cross_val_score

y = zeros(NTEMPS * NTRIALS)
idx = np.argmax(estimates, -1).reshape(-1, N+1)[...,0] == np.argmax(targets, -1).reshape(-1, COND)[..., 0]
y[idx] = 1
y = y.reshape(-1)
ss = zeros((N+1, NTEMPS, NTRIALS))
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits = 3, test_size = 0.2, random_state = 0)
cv =  LeaveOneOut()

for i in range(N + 1):
    for temp in range(NTEMPS):
        xi = estimates[[temp], :, i, maxestimates[temp, :, i]].T
        yi = equal(maxT[temp, :, 0], maxestimates[temp, :, i])
#        ss[i] = cross_val_score(clf, xi, yi, cv = cv).mean()
        for trainidx, testidx in cv.split(xi):
            try:
#                print(i, temp, yi[trainidx].mean())
                clf.fit(xi[trainidx], yi[trainidx])
                ss[i, temp, testidx] = clf.predict(xi[testidx]) == yi[testidx]
            except Exception as e:
                if all(yi[trainidx]):
                    ss[i, temp, testidx] = 1
#                print(i, temp, yi.mean(), e)
#                ss[i, temp, testidx] = 0
#                
#                ss[i, temp, testidx] = yi[trainidx].mean()

fig, ax = subplots()
ax.imshow(ss.mean(-1).T)
# %%
idx = 3
jdx = 0
true = equal(maxT[jdx, : , 0], maxC[jdx, :,idx]) 
prec = metrics.precision_score(true, array(ss[idx, jdx, :], dtype = bool))
acc = metrics.accuracy_score(true, array(ss[idx, jdx, :], dtype = bool))
print(acc, prec)
# %%
# group data by temp or by temp?
locs     = arange(NNUDGE - 1) * 2
width    = 1
tempLocs = linspace(0, 2 + N * width * 2, NTEMPS)

rcParams['axes.labelpad'] = 60
fig, ax = subplots(1, 2,  sharey = 'all')
mainax = fig.add_subplot(111, frameon = False)
mainax.set(ylabel ='prediction accuracy(%)' )

condLabels = 'Underwhelming Overwhelming'.split()
labels = 'max $\mu_i$\tdeg\tclose\tbet\tev'.split('\t')
for cond in range(COND):
    tax = ax[cond]
    tax.set_title(condLabels[cond])
    tax.set(xticks = tempLocs + .5 * 4 * width, \
            xticklabels = [f'T={round(i,2)}' for i in temps])
    tax.tick_params(axis = 'x', rotation = 45)
    for temp in range(NTEMPS):
        for ni in range(N + 1):
                x, y = tempLocs[temp], percentages[temp, ni, cond]
                x += width * ni
                y *= 100
                tax.bar(x, y, \
                        color = colors[ni], label = labels[ni])
                if y > 0:
                    tax.text(x, y + 2 , int(y), horizontalalignment = 'center',\
                             fontweight = 'bold')
    
    
tax.legend(tax.patches[:N+1], labels, loc = 'upper left', \
                              bbox_to_anchor = (1.0, 1),\
                              borderaxespad = 0, frameon = False,\
                              title = 'Method',\
                              title_fontsize = 15,) 
subplots_adjust(wspace = 0)
fig.savefig(figDir + 'statistics_overview.eps', format = 'eps', dpi = 1000)
# %%
percentages[percentages == 0] = 0
contin = scipy.stats.chi2_contingency
counts = percentages * NTRIALS
res = contin(counts, correction = True)
print(res)


# %%
conds = f'II UN OV'.split()
for i in centralities:
    conds.append(f'{i}')
gLabels = []
for temp in temps:
    for cond in conds:
        gLabels.append(f'{round(temp,2)}_{cond}') 
print(gLabels)

# %% statistical tests

methods =  'II DEG CLOS BET EV'.split()
cis     = 'UN OV'.split()
levels  = [[], [], []]

import pandas as pd
for temp in temps:
    for method in methods:
        for ci in cis:
            levels[0].append(f'T={round(temp,2)}')
            levels[2].append(method)
            levels[1].append(ci)
names = 'Temperature Method CI'.split()
index = pd.MultiIndex.from_tuples(zip(*levels), names = names)
s = pd.Series(percentages.ravel(), index = index)
print(s)
# %%
#driver = pd.DataFrame(all_drivers.T, columns = gLabels)
#print(driver)
## kruskal wallis
#results = vstack((percentage, centApprox.reshape(-1, COND)))
#
#H, p  = scipy.stats.kruskal(*results.T)
#print(f"H = {H}, p = {p:e}")
#import scikit_posthocs as sp, pandas as pd
#indx = array([1, 8, 9, 10, 11])
#
##df = pd.DataFrame(tmp, columns = "II Deg Close Betw Ev".split())
##df = df.melt(var_name = 'groups', value_name = 'values')
## %% post hoc
#df = driver.melt(var_name = 'groups')
#print(df)
#pc = sp.posthoc_conover(df, group_col = 'groups', val_col = 'value', p_adjust = 'bonferroni')
#
#heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
#fig, ax = subplots()
#sp.sign_plot(pc , **heatmap_args)
# %%
#import scikit_posthocs as sp
#percentage = percentage = array([i == target for i in rankings.T]).mean(1)
#
#test = hstack((rankings, target[:, None]))
##test2 =
#res =  scipy.stats.kruskal(*test.T)
#print(res)

 # %% compute concistency
#
#bins = arange(-.5, model.nNodes + .5)
#cons = lambda x : nanmax(histogram(x, bins , density = True)[0])
#
#ranking, maxim = stats.rankData(aucs)
#consistency = array([ [cons(i) for i in j] for j in maxim.T])
## plot consistency of the estimator
#naive = (maxim[...,0] == maxim[...,1])

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
