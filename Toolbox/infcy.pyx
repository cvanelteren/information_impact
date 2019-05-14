# cython: infer_types=True
# distutils: language=c++
# __author__ = 'Casper van Elteren'

# MODELS
# from Models.models cimport Model
from Models.models cimport Model

import numpy as np
cimport numpy as np
cimport cython
import time
from cython.parallel cimport parallel, prange, threadid
import multiprocessing as mp
import copy
from cpython.ref cimport PyObject

# progressbar
from tqdm import tqdm   #progress bar

# cython
from libcpp.vector cimport vector
from libc.stdlib cimport srand
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf
import ctypes
from cython.view cimport array as cvarray
from timeit import default_timer as timer
from cython.operator cimport dereference as deref, preincrement as prec
from cpython cimport PyObject, Py_XINCREF, Py_XDECREF
cdef extern from *:
    """
    #include <Python.h>
    #include <mutex>

    std::mutex ref_mutex;

    class PyObjectHolder{
    public:
        PyObject *ptr;
        PyObjectHolder():ptr(nullptr){}
        PyObjectHolder(PyObject *o):ptr(o){
            std::lock_guard<std::mutex> guard(ref_mutex);
            Py_XINCREF(ptr);
        }
        //rule of 3
        ~PyObjectHolder(){
            std::lock_guard<std::mutex> guard(ref_mutex);
            Py_XDECREF(ptr);
        }
        PyObjectHolder(const PyObjectHolder &h):
            PyObjectHolder(h.ptr){}
        PyObjectHolder& operator=(const PyObjectHolder &other){
            {
                std::lock_guard<std::mutex> guard(ref_mutex);
                Py_XDECREF(ptr);
                ptr=other.ptr;
                Py_XINCREF(ptr);
            }
            return *this;

        }
    };
    """
    cdef cppclass PyObjectHolder:
        PyObject *ptr
        PyObjectHolder(PyObject *o) nogil

# print gil stuff; no mp used currently so ....useless
def checkDistribution():
    '''Warning statement'''
    from platform import platform
    if 'windows' in platform().lower():
        print('Warning: Windows detected. Please remember to respect the GIL'\
              ' when using multi-core functions')
checkDistribution() # print it only once

# TODO rework this to include agentStates
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef int encodeState(long[::1] state) nogil:
    """
    Maps state to decimal number.

    NOTE this only works for binary,\
    needs to be extended to include larger than binary case
    """
    cdef:
        int binNum = 1
        int N = state.shape[0]
        int i
        int dec = 0
    for i in range(N):
        if state[i] == 1:
            dec += binNum
        binNum *= 2
    return dec

# TODO rework this to include agentStates
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef vector[long] decodeState(int dec, int N) nogil:
    """
    Decodes decimal number to state.

    NOTE this only works for binary,\
    needs to be extended to include larger than binary case
    """
    cdef:
        int i = 0
        # long[::1] buffer = np.zeros(N, dtype = int) - 1
        vector [long] buffer = vector[long](N, -1) # init with -1
    while dec > 0:
        if dec % 2:
            buffer[i] = 1
        i += 1
        dec = dec // 2
    return buffer

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef dict getSnapShots(Model model, int nSamples, int steps = 1,\
                   int burninSamples = int(1e3)):
    """
    Determines the state distribution of the :model: in parallel. The model is reset
    to random state and simulated for :step: + :burninSamples: steps after which
    a single sample is drawn and added to the output :snapshots:

    Input:
        :model: a model according to :Models.models:
        :nSamples: number of state samples to draw
        :step: number of steps between samples
    Returns:
        :snapshots: dict containing the idx of the state as keys, and probability as values
    """
    cdef:
        # unordered_map[int, double] snapshots
        # unordered_map[int, vector[int]] msnapshots
        dict snapshots = {}
        int step, sample
        int N          = nSamples * steps
        # long[:, ::1] r = model.sampleNodes(N)
        double Z       = <double> nSamples
        int idx # deprc?
        unordered_map[int, vector[int]].iterator got
        double past    = timer()
        list modelsPy  = []
        vector[PyObjectHolder] models_
        Model tmp
        cdef int tid, nThreads = mp.cpu_count()
    # threadsafe model access; can be reduces to n_threads
    for sample in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.reset()

        # TODO: remove this
        try:
            tmp.burnin(burninSamples)
        except:
            pass
        tmp.seed += sample # enforce different seeds
        # modelsPy.append(tmp)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    # init rng buffers
    cdef int sampleSize = model.nNodes if model.updateType != 'single' else 1

    cdef long[:, :, ::1] r    = np.ndarray((nThreads, steps, sampleSize), \
                                           dtype = long)
    # cdef cdef vector[vector[vector[int][sampleSize]][nTrial]][nThreads] r    = 0
    # cdef long[:, :, ::1] r = np.ndarray((nThreds, steps, sampleSize), dtype = long)
    cdef PyObject *modelptr
    pbar = tqdm(total = nSamples)
    cdef tuple state
    cdef int counter = 0
    for sample in prange(nSamples, nogil = True, \
                         schedule = 'static', num_threads = nThreads):

        tid      = threadid()
        modelptr = models_[tid].ptr
        r[tid] = (<Model> modelptr).sampleNodes(steps)
        # r[sample] = (<Model> models_[sample].ptr).sampleNodes(steps)
        # perform n steps
        for step in range(steps):
            (<Model> modelptr)._updateState(\
                                                    r[tid, step]
                                                        )
        with gil:
            state = tuple((<Model> modelptr)._states.base)
            snapshots[state] = snapshots.get(state, 0) + 1 / Z
            pbar.update(1)
    print('done')
    pbar.close()
    print(f'Found {len(snapshots)} states')
    print(f"Delta = {timer() - past: .2f} sec")
    return snapshots


cimport openmp
@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef dict monteCarlo(\
               Model model, dict snapshots,\
               int deltas = 10,  int repeats = 11,
               int nThreads = -1):
    """
    Monte-Carlo methods for estimating p(s_i^{t+\delta} | S).
    Input:
        :model: using the basic framework from Models.models
        :snapshots: dict with keys containing the state idx and the value containing its probability
        :deltas: amount of time steps to simulate
        :repeats:  number of times to repeat the simulation;
        :nThreads: number of threads to use (default = -1 : all)
    Returns:
        :conditional: dict containing the probabilities of nodes of time (number of nodes x deltas)
        for each state from :snapshots:
    """
    # TODO: solve the memory issues;  currently way too much is ofloaded to the memory
    # one idea would be to have the buffers inside the for loop. However prange is not possible
    # if that is used.
    print("Decoding..")
    # setup
    cdef:
        float past = timer()
    # pre-declaration
        double Z              = <double> repeats
        double[::1] copyNudge = model.nudges.base.copy()
        bint reset            = True
        # loop stuff
        # extract startstates
        # list comprehension is slower than true loops cython
        # long[:, ::1] s = np.array([decodeState(i, model._nNodes) for i in tqdm(snapshots)])
        long[:, ::1] s = np.array([i for i in tqdm(snapshots)])
        # long[:, ::1] s   = np.array([msnapshots[i] for i in tqdm(snapshots)])
        int states       = len(snapshots)
        # vector[int] kdxs = list(snapshots.keys()) # extra mapping idx

        # CANT do this inline which sucks either assign it below with gill or move this to proper c/c++
        # loop parameters
        int repeat, delta, node, statei, half = deltas // 2, state
        dict conditional = {}
        # unordered_map[int, double *] conditional
        long[::1] startState
        int jdx

        # long[  :,       ::1] r       = model.sampleNodes( states * (deltas) * repeats)
        # list m = []

        int nNodes = model._nNodes, nStates = model._nStates
        long[::1] agentStates = np.asarray(model.agentStates)
        str nudgeType = model._nudgeType

        unordered_map[int, int] idxer = {state : idx for idx, state in enumerate(agentStates)}

        list modelsPy = []
        vector[PyObjectHolder] models_
        Model threadModel
        PyObject *modelptr

    # setup thread count
    nThreads = mp.cpu_count() if nThreads == -1 else nThreads
    # setup output storage

    # threadsafe model access; can be reduces to n_threads
    # for state in range(nThreads):
    for state in range(nThreads):
        threadModel = copy.deepcopy(model)
        threadModel.seed += state # enforce different seeds
        # print(threadModel.t)
        # print(threadModel.nudges.base)
        # modelsPy.append(threadModel)
        models_.push_back(PyObjectHolder(<PyObject *> threadModel))


    cdef int sampleSize = model.nNodes if model.updateType != 'single' else 1

    # pre-define rng matrix
    # cdef long[:, ::1] r = model.sampleNodes(states * deltas * repeats)

    # bin matrix
    # TODO: check the output array; threading cant be performed here; susspicious of overwriting
    cdef double[:, :, :, ::1] out = np.zeros((nThreads     , deltas, \
                                              model._nNodes, model._nStates))
    cdef int nTrial = deltas * repeats
    cdef long[:, :, ::1] r = np.ndarray((nThreads, nTrial,\
                                         sampleSize), dtype = long)
    # cdef vector[vector[vector[int][sampleSize]][nTrial]][nThreads] r = 0

    pbar = tqdm(total = states) # init  progbar

    cdef int tid  # init thread id

    cdef double ZZ

    cdef Model t
    print('starting runs')
    # cdef double ZZ
    with nogil, parallel(num_threads = nThreads):
        for state in prange(states, schedule = 'static'):
            # tid         = threadid()
            tid         = openmp.omp_get_thread_num()
            modelptr    = models_[tid].ptr
            out[tid]    = 0 # reset buffer
            r[tid]      = (<Model> modelptr).sampleNodes(nTrial)
            # (<PyObject> models_[tid]).sampleNodes(1)
            for repeat in range(repeats):
                (<Model> modelptr)._states[:] = s[state, :]
                (<Model> modelptr)._nudges[:] = copyNudge[:]
                for delta in range(deltas):
                    for node in range(nNodes):
                        jdx = idxer[(<Model> modelptr)._states[node]]
                        out[tid, delta, node, jdx] += 1 / Z
                    jdx = (delta + 1) * (repeat + 1)#  * (state + 1)
                    (<Model> modelptr)._updateState(r[tid, jdx - 1])
                    if nudgeType == 'pulse' or \
                    nudgeType    == 'constant' and delta >= half:
                        (<Model> modelptr)._nudges[:] = 0
            # TODO: replace this with   a concurrent unordered_map
            with gil:
                conditional[tuple(s.base[state])] = out.base[tid].copy()
                pbar.update(1)
    pbar.close()
    print(f"Delta = {timer() - past: .2f} sec")
    return conditional

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef mutualInformation(dict conditional, int deltas, \
                          dict snapshots, Model model):
    '''
    Returns the node distribution and the mutual information decay
    '''
    cdef  np.ndarray px = np.zeros((deltas, model._nNodes, model._nStates))
    cdef  np.ndarray H  = np.zeros((deltas, model._nNodes))
    for key, p in conditional.items():
        # p    = np.asarray(p)
        H   -= np.nansum(p * np.log2(p), -1) * snapshots[key]
        px  += p  * snapshots[key] # update node distribution
    H += np.nansum(px *  np.log2(px), -1)
    return px, -H

cpdef runMC(Model model, dict snapshots, int deltas, int repeats, dict kwargs = {}):
    """ wrapper to perform MC and MI"""
    cdef:
        dict conditional = monteCarlo(model = model, snapshots = snapshots,\
                        deltas = deltas, repeats = repeats,\
                        **kwargs)
        np.ndarray px, mi
    px, mi = mutualInformation(conditional, deltas, snapshots, model)
    return conditional, px, mi
