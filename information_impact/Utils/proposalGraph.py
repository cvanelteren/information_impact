if __name__ == '__main__':
    import networkx as nx
    import fastIsing, information
    from matplotlib.pyplot import *; from numpy import *

    #fig, ax = subplots()
    close('all')
    g = nx.path_graph(2)
#    g = nx.path_graph(3, create_using = nx.DiGraph())
#    h = nx.path_graph(4, create_using = nx.DiGraph())
#    g = nx.disjoint_union(g, h)
#    g.add_edge(0, 5)
#    assert 0
    g = nx.path_graph(5, create_using = nx.DiGraph())
    h = nx.path_graph(5, create_using = nx.DiGraph())
    g = nx.disjoint_union(g, h)
    g.add_edge(0, 5)
    nx.set_edge_attributes(g, {edge : 1 for edge in g.edges()}, 'weight')

    import plotting as plotz
    plotz.showGraph(g)
#    assert 0
    #pos = {node : (i, 0) if node == 0 else (i, .01) for i, node in enumerate(g.nodes())}

    #fig, ax = subplots()
    #nx.draw(g, ax = ax, pos = pos, with_labels = g.nodes(), node_color = None)
    #savefig('Figures/test')

    temperatures = linspace(0, 10, 20)
    temperatures = hstack((0, temperatures))
    a = fastIsing.matchTemperature(g, temperatures, nSamples = 1000)

    print('test' )
    # %%
    a = fastIsing.matchTemperature(g)
    import information, fastIsing
    model = fastIsing.Ising(graph = g, temperature = 1, doBurnin = True)
    stdParams = dict(model = model,\
                    nSamples = 1000,\
                    step     = 20,\
                    deltas   = 10,\
                    repeats = 1000,\
                    updateType= 'source', reset = True)
    import time;
    past = time.time()
    results = {}
    results['control'] = information.nudgeOnNode({}, **stdParams)
    effect = 1
    # %%
    for node in g.nodes():
        nudge = {node : effect}
        results[node] = information.nudgeOnNode(nudge, **stdParams)
    print('elsaped {}'.format(time.time() - past))


    plotz.showImpact(model, results, 'all')
    # %%
    plotz.c()
    sig = lambda x, a, b :  1 / (1 + exp(a * (x - b)))
#    func = lambda x, a, b, c, d, e, f, g, h:   sig(x,a,b) * exp(c  * (x - d)) \
    # + f *tanh(- (x - e))* exp(-g * x - h)

#    i = plotz.showFit(model, results['control']['I'], func)
#    print(i[:,-1])

    func = lambda x, a, b, c:  a*exp(-b*x**c)
    func = lambda x, a, b, c: a * exp(-(x + b) * c)

    I = results['control']['I']
#    i = plotz.showFit(model, I[:,  1:], func)
#    print(i[:,-1])

    from scipy import optimize
    x = arange(len(I.T))
    xx =linspace(0, max(x), 1000)
    d = [optimize.curve_fit(func, x, y, maxfev = 10000)[0] for y in I]
    colors = cm.get_cmap('tab10')(x)
    fig, ax = subplots();
    for c, i, di in zip(colors, I, d):
        ax.scatter(x, i,  color = c)
        ax.plot(xx, func(xx, *di), '--', color = c)

#    ax = gca(); ax.set_yscale('log')
#    plotz.showFit(model, results['control']['I'], c)
