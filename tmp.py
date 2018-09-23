#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:28:22 2018

@author: casper
"""

from numpy import *
from matplotlib.pyplot import *


x = linspace(0, 100, 1000)
f, t = 50, max(x)//2
a = sin(2 * pi * f * x)  +  heaviside(x - t, 1) * f
b = (a - heaviside(x - (t + 1), 1) * f ).copy()
fig, ax = subplots()
ax.plot(x, a)
ax.plot(x, b, '--')
ax.axvline(max(x), ymin = 0, ymax = max(x), linestyle = '--', color = 'red')
ax.legend(['Nudge', 'Pulse', 'Start measuring'])
setp(ax, **dict(xlabel = 'time [step]', ylabel = 'some variable', title = \
                'Nudge vs Pulse'))