'''
    #TODO:
        - Create some basic plotting functionality, e.g.:
            - show model graph
            - Mutual information over time
            - IDT [relative absolute] && Area under the curve
'''
from numpy import *; from matplotlib.pyplot import *
import time, scipy, networkx as nx
from .stats import hellingerDistance
import scipy.optimize, scipy.integrate
# %%

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
    x   = arange(nDelta) if x is None else x # add a
    idt = zeros( (nNodes, 2) ) # store idt, rate [width at maximum height],  area under the curve, and sum squared error (SSE)
    coefficients = zeros( (nNodes, func.__code__.co_argcount - 1) )  # absolute and relatives idt
    errors       = zeros((nNodes))
    for idx, yi in enumerate(y):
        coeffs, coeffs_var = scipy.optimize.curve_fit(func, x, yi, \
                                                      **params)
#        print(coeffs, coeffs_var)
        errors[idx]        = ((func(x, *coeffs) - yi)**2).mean()
        coefficients[idx]  = coeffs

    return coefficients, errors

def extractImpact(data):
    """
    Assumes the input is a dict with keys the impact condition,
    attempts to extract the impacts
    """
    return {key : array([hellingerDistance(i.px, data['{}'].px) for i in val]) for key, val in data.items() if key != '{}'}


def c():
    close('all')




# %%
def showImpact(model, results, showBoth = 'without'):
    # TODO: do we need to include the node being nudged or not? -> only works if only 1 is nudged
    # TODO: clean this up
    '''
    Shows the impact of the node.
    Impact is defined as the sum squared difference on the node probability compared to control
    [see below]
    NOTE this assumes results has a first index 'control'
    '''
    assert len(results) > 2, 'Did you nudge the data?'

    control = results['control']['node']
    impacts = {}
    x       = arange(len(results) - 1) # for plotting  as well as indexing in 'with' case
    for resultCondition, values in results.items():
        if resultCondition != 'control':
            # look for the correct index, disregarding the index if we only want to see the effect of nudging the node
            if type(resultCondition) is int or type(resultCondition) is str:
                idx = [\
                       idx for name, idx in model.mapping.items() if idx != resultCondition\
                       ] if showBoth == 'without' else model.nodeIDs
            elif type(resultCondition) is tuple:
                idx = [\
                       idx for name, idx in model.mapping.items() if idx not in resultCondition\
                       ] if showBoth == 'without' else model.nodeIDs
            print(idx)
            nudgeNodeProbability = values['node']
            # compute hellinger distance only for the subset (else all nodes are used)
            impacts[resultCondition] = hellingerDistance(\
                   control[idx], nudgeNodeProbability[idx]).mean() # without mean it is node x delta x node state
    # PLOTTING
    fig, ax = subplots()
    # formatting
    formatting = dict(xlabel = 'nudge on', ylabel = 'impact (hellinger distance)', \
                      title = 'Impact on node distribution', xticks = x, \
                      xticklabels = list(impacts.keys()))

    ax.bar(x = x, height = list(impacts.values()), align = 'center')
    setp(ax, **formatting)
    return impacts



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

def addGraphPretty(graph, ax, \
                   positions = None, \
                   cmap      = cm.tab20, \
                   mapping   = None,\
                   **kwargs):

    from matplotlib.patches import FancyArrowPatch, Circle, ConnectionStyle, Wedge
#    graph = model.graph.

    if positions is None:
        positions = nx.circular_layout(graph)
# colors  = cm.tab20(arange(model.nNodes))
# colors  = cm.get_cmap('tab20')(arange(model.nNodes))
    from matplotlib import colors
    if isinstance(cmap, colors.Colormap):
        colors = cmap(arange(graph.number_of_nodes()))
    elif isinstance(cmap, ndarray):
        colors = cmap
    else:
        raise ValueError('Input not recognized')

    # default radius
#    r = np.array(list(positions.values()))
    # average distance
#    r = linalg.norm(r[:, None] - r[None, :], axis = 0).mean() * .1
    # DEFAULTS
    circlekwargs = dict(\
                             radius    = 10, \
                             alpha     = 1, \
                             edgecolor = 'black', \
                             linewidth = 1, \
                             linestyle = 'solid',\
                             zorder    = 2)
    for k, v in kwargs.get('circle', {}).items():
            circlekwargs[k] = v

    annotatekwargs = dict(\
                          horizontalalignment = 'center', \
                          verticalalignment   = 'center', \
                          transform           = ax.transAxes, \
                          fontsize            = circlekwargs.get('radius', 1),\
                          )

    for k, v in kwargs.get('annotate', {}).items():
        annotatekwargs[k] = v


    # get length of the fontsize
#    diamax = 6 * circlekwargs['radius']
    circlekwargs['radius'] = 1.2 * np.array([len(str(n)) for n in graph]).max()

    nodePatches = {}
    for ni, n in enumerate(graph):
        # make circle
        if mapping is not None: # overwrite default if no mapping
            ni = mapping[n]
        color = colors[ni]
        c = Circle(positions[n], \
                   facecolor = color,\
                   **circlekwargs)
        # add label
        # print(c.center[:2], pos[n])
        # ax.text(*c.center, n, horizontalalignment = 'center', \
        # verticalalignment = 'center', transform = ax.transAxes)
#        print(.95 * circlekwargs['radius'])
        annotatekwargs['fontsize'] = 8 #circlekwargs['radius'] * 8  / len(n)
        ax.annotate(n, c.center, **annotatekwargs)
        # add to ax
        ax.add_patch(c)
        # TODO : remove tis for member assignment

        # bookkeep for adding edge
        nodePatches[n] = c

    seen={} # bookkeeping
    from copy import copy
    # arrow style for drawing
    arrowsprops = dict(\
                   arrowstyle = '-|>' if graph.is_directed() else '<|-|>' ,\
                   mutation_scale = 15.0,\
                   lw = 2,\
                   alpha = 1,\
                   )
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
        if (u,v) in seen:
            rad = seen.get((u,v))
            rad = ( rad + np.sign(rad) *0.1 ) * -1

        # set properties of the edge
        alphaEdge = clip(abs(d), .2, 1)
        arrowsprops['color'] = 'green' if d > 0 else 'red'
#        arrowsprops['alpha'] = alphaEdge
        if maxWeight != minWeight:
            arrowsprops['lw'] = ((maxWeight - d) / (maxWeight - minWeight)) * 5

        # self-edge is a special case
        if u == v:
            n2 = copy(n1)
            n1 = copy(n1)
            theta     = random.uniform(0, 2*pi)
            r         = circlekwargs.get('radius')
            rotation  = pi
            corner1   = array([sin(theta), cos(theta)]) * r
            corner2   = array([sin(theta + rotation), cos(theta + rotation)]) *\
            r + .12 * sign(random.randn()) * r
            ax.annotate(*c.center)
            n1.center = array(n1.center) + corner1
            n2.center = array(n2.center) + corner2

#            n1.center = (n1.center[0] + r, n1.center[1] )
#            n2.center = (n2.center[0] - r, n2.center[1])
            rad = radians(135) # trial and error

            # add edge
            e = FancyArrowPatch(n2.center, n1.center, **arrowsprops)
            # e.set_arrowstyle(head_length = head_length)
        else:
            e  = FancyArrowPatch(n1.center,n2.center, patchA = n1, patchB = n2,\
                                connectionstyle = f'arc3,rad={rad}',
                                **arrowsprops)


        seen[(u,v)]=rad
        ax.add_patch(e)
    ax.autoscale()
    ax.set_aspect('equal')
    return ax
