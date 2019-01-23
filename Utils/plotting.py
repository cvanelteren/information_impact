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
    """Aligns colorbar with axis height"""
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, **kwargs)

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
        coeffs, coeffs_var = scipy.optimize.curve_fit(func, x, yi, **params)
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

def pt_bayescount(Pr, Nt):
    # all credit goes to panzeri-treves
    """Compute the support for analytic bias correction using the
    Bayesian approach of Panzeri and Treves (1996)

    :Parameters:
      Pr : 1D aray
        Probability vector
      Nt : int
        Number of trials

    :Returns:
      R : int
        Bayesian estimate of support

    """

    # dimension of space
    dim = Pr.size

    # non zero probs only
    PrNZ = Pr[Pr>np.finfo(np.float).eps]
    Rnaive = PrNZ.size

    R = Rnaive
    if Rnaive < dim:
        Rexpected = Rnaive - ((1.0-PrNZ)**Nt).sum()
        deltaR_prev = dim
        deltaR = np.abs(Rnaive - Rexpected)
        xtr = 0.0
        while (deltaR < deltaR_prev) and ((Rnaive+xtr)<dim):
            xtr = xtr+1.0
            Rexpected = 0.0
            # occupied bins
            gamma = xtr*(1.0 - ((Nt/(Nt+Rnaive))**(1.0/Nt)))
            Pbayes = ((1.0-gamma) / (Nt+Rnaive)) * (PrNZ*Nt+1.0)
            Rexpected = (1.0 - (1.0-Pbayes)**Nt).sum()
            # non-occupied bins
            Pbayes = gamma / xtr
            Rexpected = Rexpected + xtr*(1.0 - (1.0 - Pbayes)**Nt)
            deltaR_prev = deltaR
            deltaR = np.abs(Rnaive - Rexpected)
        Rnaive = Rnaive + xtr - 1.0
        if deltaR < deltaR_prev:
            Rnaive += 1.0
    return Rnaive

def showIDTCENT(idt, model, cent = 'betweenness'):
    from functools import partial
    # options
    cents = dict(betweenness = partial(nx.betweenness_centrality, weight = 'weight'),\
                 closeness = partial(nx.closeness_centrality, distance = 'weight'), \
                 degree_weighted = partial(nx.degree, weight = 'weight'),\
                 degree = nx.degree,\
                 current_flow = nx.current_flow_betweenness_centrality,\
                 current_flow_weighted = partial(nx.current_flow_betweenness_centrality, weight = 'weight'),\
                 communicability = nx.communicability_betweenness_centrality,\
                 eigenvector = nx.eigenvector_centrality,
                 katz = nx.katz_centrality)
    condList = 'closeness'
    if cent in condList:
        h = nx.get_edge_attributes(model.graph, 'weight')
        h = {key: abs(value) for key, value in h.items()}
        nx.set_edge_attributes(model.graph, h, 'weight')
    colors = cm.get_cmap('tab20')(arange(0, model.nNodes))
    fig, ax = subplots()

    centData = cents[cent](model.graph)
    print(centData)
    for color, (name, idx) in zip(colors, model.mapping.items()):
        ax.scatter(centData[name], idt[idx], color = color, label = name)
    setp(ax,**dict(xlabel = cent, ylabel = 'idt'))
    ax.legend()
    return centData
#%%
#from information import fit  # move this in here?
def showIDT(model, results, func = lambda x, a, b, c : exp(-(b * x)**c) ,
            condition = 'abs'):
    '''
    Produces a boxplot per conditions, it labels the outliers with the label in the graph
    : condition: determine which one to select:
            - abs   = absolute idt
            - rel   = relative idt [rate]
            - auc   = area under the curve
            - errors= fit error SSE
    '''
    selector = {name : idx for idx, name in enumerate(['abs', 'rel', 'auc', 'errors'])} # selector factors
    I = array([result['I'].T for idx, result in results.items()]) # get mutual information
    idts =  []  # TODO [change] :tmp list assignment as changes are being made to fit exponential
    # for each condition fit the exponential
    for i in I:
        idts.append(fit(i, func)[0])
    idts = array(idts)
    fig, ax = subplots();
    points = ax.boxplot(idts[..., selector[condition]].T, labels = list(results.keys()),
                        )
    # for every condition we need to show the outlier labels
    for idx, outliers in enumerate(points['fliers']):        # for some reason called fliers
        outlierValuesX, outlierValuesY = outliers.get_xdata(), outliers.get_ydata() # returns the possible outliers
        for  x, y in zip(outlierValuesX, outlierValuesY): # map back to which index this is
            errors     = (idts[idx, : , selector[condition]] - y)**2
            outlieridx = argmin(errors) # least square index find
            for name, idx in model.mapping.items():
                if idx == outlieridx:
                    print('>', idx, outlieridx)
                    ax.text(x + .1, y, name)
    ax.set_ylabel(condition)
    # show()
    return array(idts), points
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

# %%
def showGraph(graph, attr = 'weight'):
    '''
    Shows the adjacency matrix and the 'graphical layout'
    '''
    # TODO: add curved arreas
    # TODO: plot add zeros
    # ugly af
#    style.use('seaborn-white')
    fig, axes = subplots(1, 2)
    u = unique(list(nx.get_edge_attributes(graph, attr).values()))
    cmap = cm.get_cmap('viridis')
#    cmap = cm.get_cmap('viridis')(linspace(-1, 1, 20))
#    cmap[0,:] = 0
#    cmap.set_over('DarkViolet')
    # adjacency plot
    adj = nx.adjacency_matrix(graph, weight = 'weight').todense()

#    adj = (adj - adj.min()) / (adj.max() - adj.min())
#    adj[adj == 0] = nan
    cmap.set_bad('gray', .2)

    h = axes[0].imshow(\
            adj, aspect = 'auto',\
            cmap = cmap,\
            )

    cbar = colorbar(h, ax = axes[0], pad = .01)
    text(1.15, .5 , 'weight', rotation = 90,\
         horizontalalignment='center', \
         fontsize           = 20,\
         transform          = axes[0].transAxes)

    cbar.set_ticks([round(min(u),2), round(max(u), 2)])
    x = arange(len(graph.nodes()))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(graph.nodes(), ha='center', va = 'top', minor=False)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(graph.nodes(), ha = 'center', minor = False)
    axes[0].tick_params(axis='y', which='major', pad= 35)
#    cbar.set_ticklabels([])
    setp(axes[0], **dict(xlabel = 'node', ylabel = 'node'))
    fig.autofmt_xdate() # slant overlapping labels
    # there probably is a wrapper keyword for this....

    from matplotlib.patches import FancyArrowPatch, Circle, ConnectionStyle
    # graphical layout

    edges = nx.get_edge_attributes(graph, 'weight')
    negatives = {}
    positives = {}
    maxWeight = max(list(edges.values()))
    for key, value in edges.items():
        value = round(value, 3)
        edges[key] = value
        if value < 0:
            negatives[key] = value
        else:
            positives[key] = value

    r = 2
    pos = nx.circular_layout(graph, scale = 20)
    for n in graph:
        c=Circle(positions[n], radius=r,alpha=1, facecolor = 'white',\
                 edgecolor = 'black', linewidth  = 1, linestyle = 'solid')
        text(*c.center, n, horizontalalignment = 'center', verticalalignment = 'center')
        axes[1].add_patch(c)
        graph.node[n]['patch']=c
        x,y=positions[n]
#    pos = nx.circular_layout(graph, scale = .5)
#    tmp = nx.get_edge_attributes(graph, 'weight')
#    tmp = {key: abs(value) for key, value in tmp.items()}
#    nx.set_edge_attributes(graph, tmp, 'absweight')
#    pos = nx.kamada_kawai_layout(graph, weight=  'absweight', scale= 100000)
#    patches = nx.draw_networkx_nodes(graph, pos = pos, alpha = 1, node_size = 1500, edgecolors = 'black', node_color = 'white')
    seen={}
    from copy import copy
    for u, v in graph.edges():
        n1= graph.node[u]['patch']
        n2= graph.node[v]['patch']
        d = graph[u][v]['weight']
        rad=0.1
        if (u,v) in seen:
            rad=seen.get((u,v))
            rad=(rad+np.sign(rad)*0.1)*-1
            print(u,v )
        if u == v:
            j = .1
            n2 = copy(n1)
            n1 = copy(n1)
            n1.center = (n1.center[0] + r, n1.center[1] )
            n2.center = (n2.center[0] - r, n2.center[1])
#            n1.center = tuple(i - j for i in n1.center)
#            n2.center = tuple(i + j for i in n2.center)
            print(n1, n2)
            rad = 1.6
            print(n1.center, n2.center)
            e = FancyArrowPatch(n2.center, n1.center, connectionstyle = f'arc3,rad={rad}',\
                                arrowstyle = '-|>',\
                                mutation_scale = 10.0,
                                lw = 2,
                                alpha = abs(d),
                                color = 'red' if d < 0 else 'green')
            e.set_arrowstyle(arrowstyle= '-|>', head_length = 1.5)
        else:
            connectionstyle = f'arc3,rad={rad}'
            e = FancyArrowPatch(n1.center,n2.center, patchA = n1, patchB = n2,\
                                arrowstyle = '->',
                                connectionstyle = connectionstyle,
                                mutation_scale = 10.0,
                                lw=2,
                                alpha=  abs(d),
                                color= 'red' if d < 0 else 'green')

#        if (u,v) not in seen:
#            center = (n1.center + n2.center) / 2 + random.rand()
#            text(*center, round(d,2))
        seen[(u,v)]=rad
        axes[1].add_patch(e)
#    nx.draw_networkx_labels(graph, pos = pos,\
#                                          node_labels = graph.nodes())
#
#    nx.draw_networkx_edges(graph,
#                           pos,
#                           edgelist = negatives,
#                           width =  list(negatives.values()), edge_color = 'red',
#                           edge_labels = negatives, ax = axes[-1])
#    nx.draw_networkx_edges(graph,
#                           pos,
#                           edgelist = positives,
#                           width = list(positives.values()), edge_color =  'green',
#                           edge_labels = positives, ax = axes[-1]
#                           )
#    nx.draw_networkx_edge_labels(graph, pos, edge_labels = edges, ax = axes[-1], label_pos  = .5)
    axes[-1].autoscale()
    axes[-1].axis('off')

    # show()
def showRank(model, I, func):

    idts, coefs = fit(I.T, func)
    maxidt = idts[:,0].max()
    x = logspace(0, maxidt, 100)
    rank = zeros((len(idts), len(x)))
    for idx, xi in enumerate(x):
        data = array([func(xi, *coef) for coef in coefs])
        rank[:, idx] = argsort(data)
    colormap = cm.get_cm('tab20')(arange(model.nNodes))
#    fig, ax =

def showDegree(graph):
    deg = {key : value  for key, value in nx.degree(graph)}
    x  = list(deg.values())
    y  = [i / sum(x) for i in x]
    fig, ax = subplots()
    ax.plot(x, y, '.')
    ax.set_xlabel('Degree (k)')
    ax.set_ylabel('p(k)')

def showIMT(model, I):
    fig, ax = subplots()
    ax.plot(I)
    labels = zeros(model.nNodes, dtype = object)
    for key, name in model.mapping.items():
        labels[name] = key
    ax.legend(labels)
    setp(ax, **dict(xlabel = '$t - \delta$', ylabel = '$I(x_i^{t - \delta} ; X^T)$'))
    ax.grid()
    # show()

def c():
    close('all')

def showFit(model, I, \
            func = lambda x, a, b, c: a + b*exp(-c * x)  , \
            verbose = False):
    style.use('seaborn-poster') # large texts(!)
    fig, ax = subplots()
    from matplotlib import patheffects as pe
    colors = get_cmap(name = 'tab20')(arange(model.nNodes + 1))
    idts, coefs = fit(I.T, func, verbose = verbose)
    frmt = '^ s o _'.split(' ') # format for markers

    # some plotting indices
    n  = I.shape[-1]
    x  = arange(0, n)      # for plotting the observed
    num = 5000
    xx = linspace(0, n, num) if n > max(idts[:,0]) else linspace(\
                 0, max(idts[:,0]), num) # for plotting the fit
#    print(I.shape, coefs.shape, colors.shape); assert 0
    for idx, coef in enumerate(coefs):
#        name = model.rmapping[idx]
        i       = I[idx, :]

        color   = colors[idx, :]
        ax.scatter(x, i, c = color)                # observed points
        ax.plot(xx, func(xx, *tuple(coef)), c = color)     # fit

        # abs IDT
        ax.plot(idts[idx, 0], func(idts[idx, 0], *tuple(coef)),\
                marker = frmt[0], \
                c = colors[idx, :],\
                markersize = 15)
        # rel IDT
        ax.plot(idts[idx, 1],\
                func(idts[idx, 1], *tuple(coef)),\
                marker = frmt[1], c = color, \
                markersize = 15)

    # make label for every type not every plot
    delta = .5 # hide labels
    nodeLabels = [ax.plot(nan, '.', label = model.rmapping[i], c = colors[i, :])[0] for i, _ in enumerate(coefs)]
    typeLabels = [ax.scatter(-10, -10, marker = type, \
                          edgecolors = 'black', facecolor = 'black', label = l) \
                    for type, l in zip(frmt, 'abs rel observed fit'.split(' '))\
                  ]
#    fitLabel  = ax.plot(-10, -10, 'k-', label = 'fit')

    # make legends
    nodeLegend = ax.legend(handles = nodeLabels, loc = 'upper right', title = 'Nodes')
    typeLegend = ax.legend(handles = typeLabels, loc = 'upper center', title = 'Symbols')
#    fitLegend  = ax.legend(handles = fitLabel,   loc = 'center right')
    [gca().add_artist(i) for i in [nodeLegend, typeLegend]]


    formatting = dict(xlabel = '$t - \delta$', ylabel = 'Mutual information', \
                      title = 'updateType{}'.format(model.mode), \
                      ylim = [-.01, I.max() +  delta],
                      xlim = [-1, max(xx) + 4 * delta])
    setp(ax, **formatting)
    return idts, coefs
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

def addGraphPretty(model, ax, \
                   positions = None, \
                   cmap = cm.tab20, **kwargs):
    
    from matplotlib.patches import FancyArrowPatch, Circle, ConnectionStyle, Wedge
    r     = .115 # radius for circle
    graph = model.graph
    
    if positions is None:
        positions = nx.circular_layout(graph)
# colors  = cm.tab20(arange(model.nNodes))
# colors  = cm.get_cmap('tab20')(arange(model.nNodes))
    if graph.number_of_nodes() > 20:
        cmap = cm.get_cmap('gist_rainbow')

    colors = cmap(arange(model.nNodes))
    
    # DEFAULTS
    circlekwargs = dict(\
                             radius    = r, \
                             alpha     = 1, \
                             edgecolor = 'black', \
                             linewidth = 1, \
                             linestyle = 'solid',\
                             zorder    = 2)
    for k, v in kwargs.get('circle', {}).items():
            circlekwargs[k] = v
        
    annotatekwargs = dict(\
                          horizontalalignment = 'center', \
                          verticalalignment = 'center', \
                          transform = ax.transAxes, \
                          fontsize = 12,\
                          )
    
    for k, v in kwargs.get('annotate', {}).items():
        annotatekwargs[k] = v
    
    for n in graph:
        # make circle
        c = Circle(positions[n], facecolor = colors[model.mapping[n]],\
                   **circlekwargs)
        # add label
        # print(c.center[:2], pos[n])
        # ax.text(*c.center, n, horizontalalignment = 'center', \
        # verticalalignment = 'center', transform = ax.transAxes)
        annotatekwargs['fontsize'] = .95 * circlekwargs['radius']
        ax.annotate(n, c.center, **annotatekwargs)
        # add to ax
        ax.add_patch(c)
        # TODO : remove tis for member assignment

        # bookkeep for adding edge
        graph.node[n]['patch'] = c

    seen={} # bookkeeping
    from copy import copy
    # arrow style for drawing
    arrowsprops = dict(\
                   arrowstyle = '-|>' if graph.is_directed() else '<|-|>' ,\
                   mutation_scale = 15.0,\
                   lw = 2,\
                   alpha = 1,\
                   )
    
    edgesScaling = {(u, v): graph[u][v]['weight'] for u, v in graph.edges()}
    minWeight, maxWeight = min(edgesScaling.values()), max(edgesScaling.values())
    
    for u, v in graph.edges():
        n1      = graph.node[u]['patch']
        n2      = graph.node[v]['patch']
        d       = graph[u][v]['weight']
        rad     = 0.1
        if (u,v) in seen:
            rad = seen.get((u,v))
            rad = ( rad + np.sign(rad) *0.1 ) * -1
        
        # set properties of the edge
        alphaEdge = clip(abs(d), .2, 1)
        arrowsprops['color'] = 'green' if d > 0 else 'red'
#        arrowsprops['alpha'] = alphaEdge
        arrowsprops['lw'] = ((d - minWeight) / (maxWeight - minWeight)) * 5
        
        
        
        # self-edge is a special case
        if u == v:
            n2 = copy(n1)
            n1 = copy(n1)
            theta  = random.uniform(0, 2*pi)

            rotation  = pi
            corner1   = array([sin(theta), cos(theta)]) * r
            corner2   = array([sin(theta + rotation), cos(theta + rotation)]) *\
            r + .12 * sign(random.randn()) * r
            
            n1.center = array(n1.center) + corner1
            n2.center = array(n2.center) + corner2

#            n1.center = (n1.center[0] + r, n1.center[1] )
#            n2.center = (n2.center[0] - r, n2.center[1])
            rad = radians(135) # trial and error

            # add edge
            e = FancyArrowPatch(n2.center, n1.center, **arrowsprops)
            e.set_arrowstyle(head_length = head_length)
        else:
            e  = FancyArrowPatch(n1.center,n2.center, patchA = n1, patchB = n2,\
                                connectionstyle = f'arc3,rad={rad}',
                                **arrowsprops)
            

        seen[(u,v)]=rad
        ax.add_patch(e)
    ax.autoscale()
    ax.set_aspect('equal')
    return ax
