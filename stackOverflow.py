# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:33:22 2018

@author: Cas
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np

def draw_network(G,pos,ax,sg=None):

    for n in G:
        c=Circle(pos[n],radius=0.02,alpha=0.5)
        ax.add_patch(c)
        G.node[n]['patch']=c
        x,y=pos[n]
    seen={}
    for (u,v,d) in G.edges(data=True):
        n1=G.node[u]['patch']
        n2=G.node[v]['patch']
        rad=0.1
        if (u,v) in seen:
            rad=seen.get((u,v))
            rad=(rad+np.sign(rad)*0.1)*-1
        alpha=0.5
        color='k'

        e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
                            arrowstyle='-|>',
                            connectionstyle='arc3,rad=%s'%rad,
                            mutation_scale=10.0,
                            lw=2,
                            alpha=alpha,
                            color=color)
        seen[(u,v)]=rad
        ax.add_patch(e)
    return e


G=nx.MultiDiGraph([(1,2),(1,2),(2,3),(3,4),(2,4),
                (1,2),(1,2),(1,2),(2,3),(3,4),(2,4)]
                )

pos=nx.spring_layout(G)
ax=plt.gca()
draw_network(G,pos,ax)
#ax.autoscale()
#plt.axis('equal')
#plt.axis('off')
#plt.savefig("graph.pdf")
#plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import ScaledTranslation

point1 = (138.21, 19.5)
x1, y1 = point1
point2 = (67.0, 30.19)
x2, y2 = point2
size = 700

fig, ax = plt.subplots()
ax.scatter(*zip(point1, point2), marker='o', s=size)

# if I need to get and use the angles
dx = x2 - x1
dy = y2 - y1
d = np.sqrt(dx**2 + dy**2)

arrows = FancyArrowPatch(posA=(x1, y1), posB=(x2, y2),
                            color = 'k',
                            arrowstyle="-|>",
                            mutation_scale=700**.5,
                            connectionstyle="arc3")

ax.add_patch(arrows)


# shift size points over and size points down so you should be on radius
# a point is 1/72 inches
def trans_callback(event):
    dpi = fig.get_dpi()
    node_size = size**.5 / 2. # this is the radius of the marker
    offset = ScaledTranslation(node_size/dpi, -node_size/dpi, fig.dpi_scale_trans)
    shadow_transform = ax.transData + offset
    arrows.set_transform(shadow_transform)


cid = fig.canvas.mpl_connect('resize_event', trans_callback)