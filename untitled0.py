import IO, plotting as plotz, os, re
from numpy import *
from matplotlib.pyplot import *
import networkx as nx
import networkx as nx
from tqdm import tqdm_notebook as tqdm
#%load_ext autoreload
#%autoreload 2
# %%
close('all')


res = {}
d = f'{os.getcwd()}/Data/'
def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return sorted(paths, key=os.path.getctime)
d = newest(d)[-1]
use = '2018-09-11 15:35:58.931256'
#d = f'{os.getcwd()}/Data/{use}'
# d = newest(d)
print(d)

# d = f'{os.getcwd()}'
colors = cm.tab20(arange(12))
res = {}
for file in os.listdir(d):
    if file.endswith('.pickle') and not 'mags' in file:
#        print(file)
        temp  = re.search('t=\d+\.[0-9]+', file).group() # find temp
        pulse = re.search('\{.*\}', file).group()
        res[temp] = res.get(temp, ()) + ((pulse, IO.loadPickle(f'{d}/{file}')), )

        deltas = re.search('deltas=\d+', file).group()
        deltas = int(re.search('\d+', deltas).group()) + 1 # always add 1



rres = {}
for i, j in res.items():
    rres[i] = dict(j)
k = list(rres.keys())

print(k)

# %%

import scipy
from tqdm import tqdm

theta =  1e-3

for temp in k:
    try:
        current = rres[temp]
    #    d, e, f
        func = lambda x, a, b, c: a + b * exp(-c*x) #  + d * exp(-e *(x-f))
        fr   = lambda x, a, b : func(x, *a) - b # root finder


        #%matplotlib inline
        tester = {}

        for idx, (i, j) in enumerate(tqdm(current.items())):

            mi = j['mi']
            snapshots = j['snapshots']
            conditional = j['conditional']
            model = j['model']

            colors = cm.tab20(arange(model.nNodes))
            if 'H' not in globals():
                H = zeros((model.nNodes, 2))
            elif H.shape[0] != model.nNodes:
                H = zeros((model.nNodes, 2))

            #     if i == '{}':
            fig, ax = subplots()
            tester[i] = array([snapshots[ii] * jj for ii, jj in conditional.items()])
            error = 0
            for jdx, MI in enumerate(mi.T):
                x = arange(len(MI))
                a, b = scipy.optimize.curve_fit(func, x, MI, maxfev = 100000)
        #        print(a)
                l =  a[0] + theta
#                l = .5
        #        if i == '{}':
        #            print(model.rmapping[jdx], a[0], l)
        #        l = .001
            #         l = .2 * max(MI)
            #             l = 2/np.e
                r = scipy.optimize.root(fr, 0, args = (a, l) )# ,  method = 'linearmixing')
                rot = r.x
                if rot < 0:
                    rot = 0
                xx = linspace(0, 10, 1000)
                if i == '{}':
                    H[jdx, 0] = 0 if max(MI) <= l else rot
                else:
                    H[jdx, 1] += rot / model.nNodes
                ax.plot(xx, func(xx, *a), '--', color = colors[jdx], alpha = 1)
                ax.scatter(x, MI, color = colors[jdx], label = model.rmapping[jdx],\
                           alpha = 1)
                ax.scatter(rot, func(rot, *a), \
                               color = colors[jdx], marker = 's', s = 100)
                ax.set_xlim(-.2, 3)
                ax.set_ylim(-.2, 1)
                ax.set_title(f'{temp} {i} {error}')
                ax.set_ylabel('$I(x_i(t) ; X)$')
                ax.legend(bbox_to_anchor = (1., .5))
                error += ((func(x, *a) - MI)**2).sum() / model.nNodes

            if i != '{}': close()
        # %%
        #fig, ax = subplots()
        #[ax.scatter(*H[idx, :], color = colors[idx]) for idx in range(model.nNodes)]

        ii = min(H[:,0]), max(H[:, 0])
        ax.plot(ii,ii , '--k', alpha = .2)

        w = None
        w = 'weight'
    #    for node in model.graph.nodes():
    #        f = []
    #        for n in model.graph.neighbors(node):
    #            f.append(model.graph[node][n]['weight'])
    #        degs[node] = sum(f)

        #
    #    degs = dict(nx.closeness_centrality(model.graph))
        #print(degs)
        #tmp1 = degs.copy()
        #tt = abs(array(list(degs.values())))
        #m, mm = tt.min(), tt.max()
        #for i, j in tmp1.items():
        #    tmp1[i] = (j - m) / (mm - m)
        #for i, j in tmp1.items():
        #    tmp1[i] = abs(j)
        #degs = tmp1
        # %%
    #    fig, ax = subplots()
    #    for key, value in hs.items():
    #        idx = model.mapping[key]
    #        ax.scatter( degs[key], value, color = colors[idx, :])
        # %%
        from functools import partial
        centralities = dict(deg = partial(nx.degree, weight = w), \
                            betw = nx.betweenness_centrality, \
                            close = nx.closeness_centrality)
        for centr, centrality in centralities.items():
            degs = dict(centrality(model.graph))
            fig, ax = subplots()
            x = []
            for node, deg in degs.items():
                idx = model.mapping[node]
                ax.scatter(deg, H[idx, 0], color = colors[idx], label = node)
                x.append((deg, H[idx, 0]))

            x = array(x)
            xx = scipy.stats.linregress(*x.T)
            #     ax.scatter(*H[idx, :], color = colors[idx])
            #ax.set_yscale('log')
            setp(ax, **dict(xlabel = centr, ylabel = 'abs IDT'))
            ax.legend(bbox_to_anchor = (1.01, 1))
            ax.set_title(f'T = {temp}')

        show()
        # %%
        #fig, ax = subplots()
        #for i, j in degs.items():
        #    idx = model.mapping[i]
        #    ax.scatter(j, mi[0, idx])
        #setp(ax, **dict(xlabel = 'degree', ylabel = 'entropy'))
        ## %%
        # %%
        import re

        hd = lambda x, y : sqrt(((sqrt(x) - sqrt(y))**2).sum(-1))/sqrt(2)
        iidx = 1
        control = pxs['{}']
        hs = {}
        hd_single = zeros((model.nNodes, deltas, model.nNodes))
        for i, j in current.items():
            if i != '{}':
                px = j['px']
    #            title = re.search("'.*'", i).group()[1:-1]; # print(title)
                title = re.search('{.*:', i).group()[1:-1]
                h = hd(control, px)
                try:
                    idx = model.mapping[title]
                except:
                    idx = model.mapping[int(title.split("'")[0])]
                hd_single[idx, ...] = h
                hs[title] = h.mean()




        # %%
        fig, ax = subplots()

        [ax.plot(i, color = colors[idx], label = model.rmapping[idx]) for idx, i in enumerate(hd_single.mean(axis = -1))]
        ax.legend(bbox_to_anchor = (1, 1))
        setp(ax, **dict(xlabel = 'Time', ylabel = 'Hellinger distance'))
        # plot impact
    #    fig, ax = subplots(figsize = (10, 10))
    #    ax.bar(list(hs.keys()), list(hs.values()))
    #    ax.set_ylabel('Hellinger distance')
    #

        # plot idt vs impact
        fig, ax = subplots()
        xx = []
        for node in model.graph.nodes():
            try:
                try:
                    x = hs[int(node)]
                except:
                    x = hs[str(node)]

            except:
                x = hs[f"'{node}'"]
            y = H[model.mapping[node], 0]
            xx.append((x,y))
            ax.scatter(x, y, color = colors[model.mapping[node]], label = node)

        #ax.set_yscale('log')
        xx = array(xx)
        r =  scipy. stats.linregress(xx)
        x = linspace(min(xx[:, 0]), max(xx[:, 0]))
        ax.plot(x, r.slope * x + r.intercept, 'k--')
        ax.text(.6, .2, f'p = {r.pvalue:.2f}', transform = ax.transAxes)
        setp(ax, **dict(xlabel = 'impact', ylabel = 'idt', title = f'T = {temp}'))
        ax.legend(bbox_to_anchor = (1.01, 1))
        savefig(f'{d}/{temp}.png')
        # %%

        # calc idt of hellinger
        xr = linspace(0, 10)
        fig, ax = subplots()
        HHH = zeros(model.nNodes)
        for idx, i in enumerate(hd_single.mean(axis = -1)):
            node = model.rmapping[idx]
            ii =  deltas//2 + 1
            x = arange(len(i) - ii)

            a, b = scipy.optimize.curve_fit(func, x, i[ii:],  maxfev = 1000000)
            tmp = a[0] + theta
            h_idt = scipy.optimize.root(fr, -1, args = (a, tmp), method = 'linearmixing')
            rot = h_idt.x
            if max(i) <= tmp or rot < 0:
                rot = 0
            idt = H[idx, 0]
            ax.plot(xr, func(xr, *a), color = colors[idx])
            ax.scatter(x, i[ii:], color = colors[idx], alpha = 1, label = node)
            ax.scatter(rot, func(rot, *a), 200, color = colors[idx],  marker = 's')

            HHH[idx] = rot
    #        print(idx, node, rot, max(i), tmp)

        ax.legend(bbox_to_anchor = (1.01, 1))
        ax.set_xlim(0, deltas//2)
#        ax.set_ylim(-.2, 1)
        setp(ax, **dict(xlabel = 'time since nudge', ylabel = 'hellinger distance'))

        # centrality measure and impact
        for centr, centrality in centralities.items():
            degs = dict(centrality(model.graph))
            fig, ax = subplots()
            x = []
            for node, deg in degs.items():
                idx = model.mapping[node]
                ax.scatter(deg, HHH[idx], color = colors[idx], label = node)
                x.append((deg, HHH[idx]))

            x = array(x)
            xx = scipy.stats.linregress(*x.T)
            #     ax.scatter(*H[idx, :], color = colors[idx])
            #ax.set_yscale('log')
            setp(ax, **dict(xlabel = centr, ylabel = 'impact'))
            ax.legend(bbox_to_anchor = (1.01, 1))
            ax.set_title(f'T = {temp}')

        fig, ax = subplots()
        [ax.scatter(x, y, color = colors[idx], label = model.rmapping[idx]) for idx, (x, y) in enumerate(zip(H[:,0], HHH))]
        ax.legend(bbox_to_anchor = (1.01, 1))
        ax.set_title(temp)
        setp(ax, **dict(xlabel = 'abs idt', ylabel = 'hellinger limit'))
        r = scipy.stats.linregress(H[:, 0], HHH)
        tmpx = linspace(min(H[:,0]), max(H[:,0]))
        tmpy = tmpx * r.slope + r.intercept
        ax.plot(tmpx, tmpy, 'k--')
        ax.text(.8, .1, f'p={r.pvalue:.2f}', transform = ax.transAxes)
        # %%

        conditionals = {i : j['conditional'] for i, j in current.items()}
        control = conditionals['{}']
        fig, ax = subplots()

        HH = zeros(model.nNodes)
        for i, conditional in conditionals.items():
            if i != '{}':
                title = re.search('{.*:', i).group()[1:-1].split("'")[0]
                title  = int(title)
    #            print(title)
                tmp = []
                for x, y in conditional.items():
    #                y *= snapshots[x]

                    xx = control[x]
    #                xx*=snapshots[x]
                    tmp.append(hd(xx, y))
                tmp = array(tmp)
                tmp = tmp.mean(axis = 0).mean(axis =-1)
                idx = model.mapping[title]

                ii = deltas // 2 + 1
                xxx = arange(len(tmp) - ii)
                a, b  = scipy.optimize.curve_fit(func, xxx, tmp[ii:], maxfev= 10000)
                l = a[0] + theta
                root = scipy.optimize.root(fr, 0, args = (a, l))
                ax.scatter(root.x, func(root.x, *a), c = colors[idx])
                ax.plot(xxx, tmp[ii:], color = colors[idx], label = title)
                HH[model.mapping[title]] = root.x
        fig, ax = subplots()
        [ax.scatter(x,y, c = c) for x,y,c in zip(H[:, 0], HH, colors)]
        ax.legend()
    except Exception as e: print(e)

    show()
