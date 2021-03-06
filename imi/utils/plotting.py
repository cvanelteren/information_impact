'''
    #TODO:
        - Create some basic plotting functionality, e.g.:
            - show model graph
            - Mutual information over time
            - IDT [relative absolute] && Area under the curve
'''
import numpy as np;
import matplotlib.pyplot as plt
import time, scipy, networkx as nx
import scipy.optimize, scipy.integrate
# %%
def gen_main(fig, position = (1,1,1)):
    mx = fig.add_subplot(*position, frameon = 0,\
                         xticks = [],\
                         yticks = []
                )
    return mx
def get_linear_cmap(n, name = 'nipy_spectral'):
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap(name)
    idx = np.linspace(0, 256, n, dtype = int)
    return cmap(idx)

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N, 0))
    cmap_name = base.name + str(N)
    return plt.cm.colors.ListedColormap(color_list, color_list, N)

def get_shells(graph):
    shells = {}
    for k, v in dict(graph.degree()).items():
        shells[v] = shells.get(v, []) + [k]
    return dict(sorted(shells.items(), key = lambda x: x[0])[::-1])

def get_shell_layout(graph, **kwargs):
    shells = get_shells(graph)
    return nx.shell_layout(graph, nlist = list(shells.values()), **kwargs)


def colorbar(mappable, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """Aligns colorbar with axis height"""
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax = cax)
    return cbar
#return fig.colorbar(mappable, cax=cax, **kwargs)

def fit(y, func, x = None,\
                   params = {}):
    '''
    Wrapper for scipy curve fit
    Input :
        :y: 2x matrix where time is the first dimension
        :func: function to fit
        :x: 1d array for the time, default is arange of the first dimension of :y:
        :params: options for optimize.curve_fit
    Returns:
        :coefficients: the fitted parameters of the function
        :errors: the standard deviation of the variance of the parameters
    '''
    #TODO; parallize; problem : lambda -> use pathos or multiprocessing?
    nNodes, nDelta = y.shape
    x   = np.arange(nDelta) if x is None else x # add a
    idt = np.zeros( (nNodes, 2) ) # store idt, rate [width at maximum height],  area under the curve, and sum squared error (SSE)
    coefficients = np.zeros( (nNodes, func.__code__.co_argcount - 1) )  # absolute and relatives idt
    errors       = np.zeros((nNodes))
    for idx, yi in enumerate(y):
        coeffs, coeffs_var = scipy.optimize.curve_fit(func, x, yi, \
                                                      **params)
#        print(coeffs, coeffs_var)
        errors[idx]        = ((func(x, *coeffs) - yi)**2).mean()
        coefficients[idx]  = coeffs

    return coefficients, errors

def c():
    close('all')

#close('all');
#showGraph(model.graph)
def saveAllFigures(useLabels = False, path = '../Figures'):
    '''
    Saves all open figures
    input:
        :useLabels: store in format figure [format] if useLabels is activated
        it will use the labels given to the figure
    '''
    for figNum in get_fignums():
        figure(figNum) # activate current figure
        saveStr = '{}/figure {}'.format(path, get_figLabels()[figNum] if useLabels else figNum)
        print('saving at {}'.format(saveStr))
        savefig(saveStr)

def addGraphPretty(graph, \
                   ax, \
                   positions = None, \
                   cmap      = plt.cm.tab20, \
                   mapping   = None,\
                   **kwargs):
    """Short summary.

    :param graph: Description of parameter `graph`.
    :param ax: Description of parameter `ax`.
    :param : Description of parameter ``. Defaults to None.
    :param : Description of parameter ``. Defaults to cm.tab20.
    :param : Description of parameter ``. Defaults to None.
    :param : Description of parameter ``.
    :return: Description of returned object.

    """
    from matplotlib.patches import FancyArrowPatch, Circle, ConnectionStyle, Wedge
#    graph = model.graph.
    # if default
    if positions is None:
        positions = nx.circular_layout(graph)
    # check if it is callable
    elif hasattr(positions, '__call__'):
        print('User provided position function')
        positions = positions(graph)

# colors  = cm.tab20(arange(model.nNodes))
# colors  = cm.get_cmap('tab20')(arange(model.nNodes))
    from matplotlib import colors
    if isinstance(cmap, colors.Colormap):
        colors = cmap(np.arange(graph.number_of_nodes()))
    elif isinstance(cmap, np.ndarray):
        colors = cmap
    else:
        raise ValueError('Input not recognized')

    # default radius
#    r = np.array(list(positions.values()))
    # average distance
#    r = linalg.norm(r[:, None] - r[None, :], axis = 0).mean() * .1
    # DEFAULTS
    # get min distance between nodes; use that as baseline for circle size
    layout = kwargs.get('layout', {})
    s      = layout.get('scale', None)
    if s:
        positions = {i : np.array(j) * s for i, j in positions.items()}
    from scipy.spatial.distance import pdist, squareform
    tmp = np.array(list(positions.values()), dtype = float)
    s   = pdist(tmp).min() * .40 # magic numbers galore!
    circlekwargs = dict(\
                             radius    = s, \
                             alpha     = 1, \
                             edgecolor = 'none', \
                             linewidth = 1, \
                             linestyle = 'solid',\
                             zorder    = 2)
    for k, v in kwargs.get('circle', {}).items():
            circlekwargs[k] = v

    arrowsprops = dict(\
                   arrowstyle = '-|>' if graph.is_directed() else '<|-|>' ,\
                   mutation_scale = 5.0,\
                   lw = 2,\
                   alpha = 1,\
                   )
    for k, v in kwargs.get("edges", {}).items():
        arrowsprops[k] = v

    # get length of the fontsize
    #    diamax = 6 * circlekwargs['radius']


    # default text stuff
    annotatekwargs = dict(\
                          horizontalalignment = 'center', \
                          verticalalignment   = 'center', \
                          transform           = ax.transAxes, \
                          )
    for k, v in kwargs.get('annotate', {}).items():
        annotatekwargs[k] = v
    # clip at minimum string length
    tmp   = np.array([len(str(n)) for n in graph]).max()
    maxFS = tmp
    annotatekwargs['fontsize'] =  circlekwargs.get('radius') * annotatekwargs.get('fontsize', 1)
    # positions   = {i : array(j) * circlekwargs['radius'] * 4 for i,j in positions.items()}
    nodePatches = {}
    for ni, n in enumerate(graph):
        # make circle
        if mapping is not None: # overwrite default if no mapping
            ni = mapping[n]
        color = colors[ni]
        c = Circle(positions[n], \
                   facecolor = color,\
                   label = n,\
                   **circlekwargs, \
                   )
        # add label
        # print(c.center[:2], pos[n])
        # ax.text(*c.center, n, horizontalalignment = 'center', \
        # verticalalignment = 'center', transform = ax.transAxes)
#        print(.95 * circlekwargs['radius'])
        # annotatekwargs['fontsize'] = 8 #circlekwargs['radius'] * 8  / len(n)
        if annotatekwargs.get('annotate', True):
            # bbox_props = dict(boxstyle="circle", fc = 'none', ec = colors[ni], lw = 1)
            ax.annotate(n, c.center, **{k: v for k, v in annotatekwargs.items() if k != 'annotate'})
        # add to ax
        ax.add_patch(c)
        # TODO : remove tis for member assignment

        # bookkeep for adding edge
        nodePatches[n] = c

    seen = {} # bookkeeping
    from copy import copy
    # arrow style for drawing

    edgesScaling = {}
    for u, v in graph.edges():
        edgesScaling[(u, v)] = dict(graph[u][v]).get('weight', 1)
    if edgesScaling:
        minWeight, maxWeight = min(edgesScaling.values()), max(edgesScaling.values())
    else:
        minWeight, maxWeight = 0, 1

 
    for u, v in graph.edges():
        n1      = nodePatches[u]
        n2      = nodePatches[v]
        d       = dict(graph[u][v]).get('weight', 1)
        rad     = 0.1

        # no clue why i did this
        if (u,v) in seen:
            rad = seen.get((u,v))
            rad = ( rad + np.sign(rad) *0.1 ) * -1
        # set properties of the edge
        alphaEdge = np.clip(abs(d), .2, 1)
        
        arrowsprops['color'] = 'green'
        if d < 0:
            arrowsprops['color'] = 'red'
       # arrowsprops['alpha'] = alphaEdge
        if maxWeight != minWeight:
            arrowsprops['lw'] = np.clip(((maxWeight - d) / (maxWeight - minWeight)) * 2.5, .1, None) # magic numbers galore!
        # self-edge is a special case
        if u == v:
            offset = np.array(n1.center, dtype = float)

            # add * for self-edges
            offset[0] +=  1.05 * n1.radius
            if annotatekwargs.get('selfEdges', True):
                ax.annotate('*', offset, fontsize = 30)
#            n1.center = (n1.center[0] + r, n1.center[1] )
#            n2.center = (n2.center[0] - r, n2.center[1])
            rad = np.radians(135) # trial and error

            # add edge
            # e.set_arrowstyle(head_length = head_length)
        else:
            e  = FancyArrowPatch(n1.center,n2.center, patchA = n1, patchB = n2,\
                                connectionstyle = f'arc3,rad={rad}',
                                **arrowsprops)

            ax.add_patch(e)

        seen[(u,v)]=rad
    ax.autoscale()
    ax.set_aspect('equal')
    return ax


def get_main(fig, position = (1, 1, 1)):
    mx = fig.add_subplot(*position,\
                    frameon = [],
                    xticks = [],
                   yticks = [])
    return mx
