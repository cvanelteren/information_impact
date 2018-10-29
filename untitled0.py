import IO, plotting as plotz, os, re
from numpy import *
from matplotlib.pyplot import *
import networkx as nx, plotting
from tqdm import tqdm_notebook as tqdm
#%load_ext autoreload
#%autoreload 2
# %%
close('all')


res = {}
dataPath = f'{os.getcwd()}/Data/'

import scipy
from tqdm import tqdm

theta =  1e-1
METHOD = 'hybr'
func = lambda x, a, b, c, d, e, f: a + b * exp(-c* x) # + d * exp(- e * (x - f))
paths = IO.newest(dataPath)[-2]
tmp = paths if type(paths) is list else [paths]
for d in tmp:
    if not os.path.exists(d + '/figures/'):
        print("making directory")
        os.mkdir(d + '/figures/')
    deltas = IO.readSettings(d)['deltas'] + 1
    data   = IO.extractData(d)
    print(IO.readSettings(d))
    for temp in data.keys():
        try:
            current = data[temp]
        #    d, e, f
            fr   = lambda x, a, b : func(x, *a) - b # root finder


            #%matplotlib inline
            tester = {}

            for idx, (i, j) in enumerate(tqdm(current.items())):

                mi          = j.mi
                snapshots   = j.snapshots
                conditional = j.conditional
                model       = j.model


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
                    l =  a[0] + theta  * max(MI)
                    rot = log((l - a[0])/a[1]) * -1/a[2]
                    rot = 0 if isnan(rot) else rot
                    rot = 0 if rot < 0 else rot
                    ddd = rot
                    k = -1
                    while True:
                        if k > 100:
                            print(f"after {k} tries giving up")
                            break
                        r = scipy.optimize.root(fr, k, args = (a, l), method = METHOD )# ,  method = 'linearmixing')
                        if r.success:
                            break
                        k += .1
                    rot = r.x
                    if rot < 0 or max(MI) < l:
                        rot = 0
                    print(ddd, rot)
                    xx = linspace(0, 10, 1000)
                    if i == '{}':
                        H[jdx, 0] = 0 if max(MI) <= l else rot
                    else:
                        H[jdx, 1] += rot / model.nNodes
                    ax.plot(xx, func(xx, *a), '--', color = colors[jdx], alpha = 1)
                    ax.scatter(x, MI, color = colors[jdx], label = model.rmapping[jdx],\
                               alpha = 1)
                    ax.scatter(rot, func(rot, *a), \
                                   color = colors[jdx], marker = 's', s = 100, alpha = .3)
                    ax.set_xlim(-.2, 3)
                    ax.set_ylim(-.2, 1)
                    ax.set_title(f'{temp} MSE = {error:.2f}')
                    ax.set_ylabel('$I(x_i(t) ; X)$')
                    ax.set_xlabel('Time [step]')
                    ax.legend(bbox_to_anchor = (1., .5))
                    error += ((func(x, *a) - MI)**2).sum() / model.nNodes

                if i != '{}': close()

 
            savefig(f'{d}/figures/idt_time.png')
            fig, ax = subplots()
#            plotting.addGraphPretty(model, ax[0])
            ax.hist(dict(model.graph.degree()).values(), 10, density = True)
            setp(ax, **dict(xlabel = 'Degree (k)', ylabel = 'p(k)'))
            
            savefig(f'{d}/figures/degree.png')
            
            fig, ax = subplots()
            plotting.addGraphPretty(model, ax)
            savefig(f'{d}/figures/graph.png')
            # %%
            ii = min(H[:,0]), max(H[:, 0])
            ax.plot(ii,ii , '--k', alpha = .2)

            w = None
            w = 'weight'

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
                ax.set_title(f'{temp}')

            # %%
            #fig, ax = subplots()
            #for i, j in degs.items():
            #    idx = model.mapping[i]
            #    ax.scatter(j, mi[0, idx])
            #setp(ax, **dict(xlabel = 'degree', ylabel = 'entropy'))
            ## %%
            pxs = {}
            for i, j in current.items():

                pxs[i] = j.px
            # %%
            import re
            from stats import hellingerDistance as hd
            # hd = lambda x, y : sqrt((( sqrt(x) - sqrt(y) )**2).sum(-1))/sqrt(2)
            iidx = 1
            control = pxs['{}'][iidx, ...]
            hs = {}
            hd_single = zeros((model.nNodes, deltas, model.nNodes))
            for i, j in pxs.items():
                if i != '{}':
        #            title = re.search("'.*'", i).group()[1:-1]; # print(title)
                    title = re.search('{.*:', i).group()[1:-1]
                    h = hd(control, j[iidx, ...])
                    try:
                        idx = model.mapping[title]
                    except:
                        idx = model.mapping[int(title.split("'")[0])]
                    hd_single[idx, ...] = hd(pxs['{}'], j)
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
            savefig(f'{d}/figures/impact_idt.png')
            # %%

            # calc idt of hellinger
            xr = linspace(0, 10, 1000)
            fig, ax = subplots()
            HHH = zeros(model.nNodes)
            for idx, i in enumerate(hd_single.mean(axis = -1)):
                node = model.rmapping[idx]
                ii =  deltas//2
                x = arange(len(i) - ii)

                a, b = scipy.optimize.curve_fit(func, x, i[ii:],  maxfev = 1000000)
                l = a[0] + theta * max(i)
                rot = log((l - a[0])/a[1]) * -1/a[2]
                rot = 0 if isnan(rot) else rot
                rot = 0 if rot < 0 else rot
#                k = 0
#                while True:
#                    if k > 100:
#                        print(f"after {k} tries giving up")
#                        break
#                    h_idt = scipy.optimize.root(fr, k, args = (a, tmp), method = METHOD )# ,  method = 'linearmixing')
#                    if h_idt.success:
#                        break
#                    k += 1
#                if k > 0 : print(k)
#                rot = h_idt.x
#                if max(i) <= tmp or rot < 0:
#                    rot = 0
                idt = H[idx, 0]
                ax.plot(xr, func(xr, *a), alpha = .2, color = colors[idx])
                ax.scatter(x, i[ii:], color = colors[idx], alpha = 1, label = node)
                ax.scatter(rot, func(rot, *a), 100, color = colors[idx],  marker = 's')

                ax.vlines(rot, 0, func(rot, *a), color = colors[idx])
                HHH[idx] = rot
        #        print(idx, node, rot, max(i), tmp)

            ax.legend(bbox_to_anchor = (1.01, 1))
            ax.set_xlim(-.2, 3) # deltas//2)
            ax.set_ylim(- hd_single.mean(axis = -1).max() * 1.1, hd_single.mean(axis = -1).max() * 1.1)
            setp(ax, **dict(xlabel = 'time since nudge', ylabel = 'hellinger distance'))
            savefig(f'{d}/figures/impact_time.png')
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
            savefig(f'{d}/figures/idt_impact.png')
            # %%

            conditionals = {i : j.conditional for i, j in current.items()}
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

                    ii = deltas // 2
                    xxx = arange(len(tmp) - ii)
                    a, b  = scipy.optimize.curve_fit(func, xxx, tmp[ii:], maxfev= 10000)
                    l = a[0] + theta * max(tmp)
                    root = scipy.optimize.root(fr, 0, args = (a, l))
                    ax.scatter(root.x, func(root.x, *a), color = colors[idx])
                    ax.plot(xxx, tmp[ii:], '.', color = colors[idx], label = title)
                    ax.plot(linspace(0, 100), func(linspace(0, 100), *a), color = colors[idx])
                    HH[model.mapping[title]] = root.x
            ax.set_xlim(0, 4)
            fig, ax = subplots()
            [ax.scatter(x,y, color = c) for x,y,c in zip(H[:, 0], HH, colors)]
            ax.legend()
        except Exception as e: 
            print(e)
            from time import sleep; sleep(1)

        show()
