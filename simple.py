y#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:31:36 2018

@author: casper
"""
# from matplotlib.pyplot import *
from numpy import *
from scipy import optimize
import IO, os, plotting as plotz
from matplotlib.pyplot import *
#style.use('seaborn-poster')


dataPath = f"{os.getcwd()}/Data/"
tmp      = IO.newest(dataPath)[-1]
#tmp      = dataPath + '1539187363.9923286'
extractThis = f"{dataPath}{tmp}"
print(extractThis)

data   = IO.extractData(tmp)
func   = lambda x, a, b, c: a + b * exp(-c * x)  # + d * exp(-e * (x - f))
# thetas = [10**-i for i in range(1, 20)]
thetas = [0.5,  .1, .01, .001]
t       = next(iter(data))
model   = data[t]['{}'][0].model

mis = array([sample.mi for sample in data[t]['{}']])
mus = mis.mean(axis = 0)  #average over samples
stds= mis.std(axis = 0) * 1.96
x   = arange(0, mus.shape[0]) #deltas
colors = cm.tab20(arange(mus.shape[1]))

fitx  = linspace(0, x.max(), 1000)
func = lambda x, a, b, c, d, e, f, g: a + b * exp( - c * (x) )  + d * exp(- e * (x-f))
p0 = ones((func.__code__.co_argcount - 1))
p0[0] = 0
coeffs, errors = plotz.fit(mus, func, params = dict(p0 = p0, maxfev = 100000))

thetas = [.5, .1, .01, .001, .0001]

roots  = zeros((mus.shape[1], len(thetas)))
fig, ax = subplots()
for idx, (muNode, stdNode, color) in enumerate(zip(mus.T, stds.T, colors)):
    ax.plot(x, muNode,'--.', color = color, label = idx, alpha = .4, markersize = 15)
    ax.plot(fitx, func(fitx, *coeffs[idx]), color = color)
    ax.fill_between(x, muNode + stdNode, muNode - stdNode, color = color, alpha = .1)

    root = array([optimize.fsolve(lambda x: func(x, *coeffs[idx]) -  theta * muNode.max() + coeffs[idx, 0], 0)\
                        for theta in thetas])

    roots[idx] = root.squeeze()
#     print(errors[idx])
#     ax.scatter(root, func(root, *coeffs[idx]), color = color, marker = 's')
    
ax.legend(bbox_to_anchor = (1.15, 1), title = 'Node')
setp(ax, **dict(\
#                xscale = 'log',\
#               yscale = 'log',\
               xlabel = 'Time[step]',\
               ylabel = r'$I( x_i (t) ; X )$ ',
               title  = 'Observed data')
    );

print(roots.shape)
fig, ax = subplots()
h = ax.imshow(roots.argsort(axis = 0))
setp(ax, **dict(\
             xlabel = 'Theta',\
             ylabel = 'Ranking',\
             )
             )
colorbar(h, ax = ax, label = 'Node')

fig, ax = subplots()
ax.semilogy(mus.std(axis = 1))
setp(ax, **dict(\
               xlabel = 'Time[step]',\
               ylabel = 'standard deviation',\
               title  = 'Standard deviation observed values', \
               )\
    );


# %%
from mpl_toolkits.mplot3d import Axes3D
control = data[t]['{}']
impacts = {}
fig, ax = subplots(3)

rooting = zeros((len(control), len(thetas), 2))
for key, v in data[t].items():
    if key != '{}':
        idx = int(re.search('\d+', key).group())
        idx = model.mapping[idx]
        impact = array([plotz.hellingerDistance(i.px, j.px)  for i, j in zip(v, control)])
        
#        for tdx, theta in enumerate(thetas):
        print(impact.shape)
        impacts[key] = impact
        tmp = impacts[key].sum(-1)
        
        n   = tmp.shape[-1]
        x   = tmp[:, n // 2:]
        y   = mis[:, :n // 2, idx]
        
        ff = median
#        ax.scatter(ff(y, axis = 0), ff(x, axis = 0), color = colors[idx], alpha = .5)
#         print(ff(y, axis =0)[0], ff(x, axis = 0)[0])
        [ax[0].scatter(yi, xi, color = colors[idx], alpha = .6 )for jdx, (xi, yi) in enumerate(zip(x.T, y.T))]
        ax[1].plot(x.T,'--.', color = colors[idx])
        ax[2].plot(y.T,'--.', color = colors[idx])
ax[0].set_yscale('log'); ax[0].set_xscale('log')

ax[0].set_xlabel('mi')
ax[0].set_ylabel('impact')

ax[1].set_ylabel('impact')

ax[2].set_ylabel('mi')
#ax[2].set_xlim(0, 3)
#ax[1].set_xlim(0, 3)

tight_layout(pad = 0)
show()