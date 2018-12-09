#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 09:12:46 2018

@author: casper
"""

import test
import numpy as np
from copy import deepcopy
a  = test.Test(np.zeros(10))

b  = deepcopy(a)
b.states += 2
print(a.states)
print(id(b.states) == id(a.states))
del a
print(b.states)