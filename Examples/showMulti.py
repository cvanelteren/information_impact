#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:08:15 2018

@author: casper
"""

from numpy import *
from matplotlib.pyplot import *


def func(xt, a, b, c, d):
    x, y = xt
    return (a * exp(-b*x - c*y - d)).ravel()

t = linspace(0, 1)


x, y = meshgrid(t,t)
est = func((x,y), 1, 1, 1, 1)
estn = est + random.rand(*est.shape)
from scipy import optimize

d = optimize.curve_fit(func, xdata = (x,y), ydata = est)[0]
r = func((x,y), *d)

close('all')
fig, ax = subplots(2)
n = len(t)
ax[0].imshow(est.reshape(n,n))
ax[1].imshow(r.reshape(n,n))

