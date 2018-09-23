#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:35:11 2018

@author: casper
"""

from numpy import *
from matplotlib.pyplot import *
close('all')




x = linspace(0, 20, 1000)

offset = .5
std    = heaviside(-(x - offset), 1)
stdinv = heaviside(x - offset, 1)
decay  = exp(-.5 * (stdinv * x - offset))


fig, ax = subplots()
ax.plot(x, std + decay)
ax.plot(x, std + decay + .2)
ax.plot(x, std + decay + heaviside(x - offset + .05 , 1) * .2, '--')
setp(ax, **dict(xlabel = 'time', ylabel ='some variable'))
ax.legend('Normal Nudge Pulse'.split(' '))
ax.set_title('Schematic Pulse explanation')