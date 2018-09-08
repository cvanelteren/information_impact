from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from numpy import *


fig, ax = subplots(subplot_kw = dict(projection = '3d'))
d = random.rand(100,3)
print(d.shape)
ax.plot(xs = d[:,0], ys = d[:,1], zs = d[:,-1], marker = '.', linestyle = '')
show()
x
