import multiprocess as mp
from numpy import *
import h5py



def main(s, fileName = 'test'):
    # x = random.rand(
    counter = 0
    x = memmap('test.dat', dtype = 'float64', shape = s, mode = 'write')
    # f.create_group('test')
    print(x.shape)
    with mp.Pool(processes = mp.cpu_count()) as p:
        res = p.imap(calc, chunk(x))
        for start, stop, i in res:
            # print(i)
            x[start : stop, :] = i
            counter += 1
    print('Done')
    import os
    os.remove('test.dat')


def chunk(x, cs = .05):
    n = len(x)
    c = int(cs * n)
    d, r = divmod(n, c)
    interval  = arange(0, n,  c)
    t = c if r == 0 else r
    interval = hstack((interval, max(interval) + t))
    print(interval)
    for start, stop in zip(interval[:-1], interval[1:]):
        yield (start, stop, x[start:stop, ...])
def calc(y):
    start, stop, x = y
    return (start, stop, x + random.rand(*x.shape) )

def main2(s):
    x = zeros(s)
    with mp.Pool(processes = mp.cpu_count()) as p:
        res = p.map(calc, chunk(x))
if __name__ == '__main__':
    ss =(100000000, 3)
    import time
    s = time.time()
    main(ss)
    print(time.time() - s)
    s = time.time()
    main2(ss); print(time.time() - s)
