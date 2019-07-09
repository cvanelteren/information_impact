import multiprocessing as mp, numpy as np
m = mp.Manager()

N = 5
# settings = [[] for i in range(N)] 
settings = m.list()
def f(x, settings):
    settings += [x] 
    print(np.random.get_state())
if __name__ == "__main__":
    from functools import partial
    tmp = partial(f, settings = settings)
    with mp.Pool(mp.cpu_count()) as p:
        p.map(tmp, range(N))
