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
        unordered_map[int, double] snapshots
        int step, sample
        int N          = nSamples * steps
        # long[:, ::1] r = model.sampleNodes(N)
        double Z       = <double> nSamples
        int idx
        double past    = timer()
        list modelsPy  = []
        vector[PyObjectHolder] models_
        Model tmp
        cdef int tid, nThreads = mp.cpu_count()
    # threadsafe model access; can be reduces to n_threads
    for sample in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.reset()
        tmp.burnin(burninSamples)
        tmp.seed += sample # enforce different seeds
        modelsPy.append(tmp)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))

    # init rng buffers
    cdef int sampleSize = model.nNodes if model.updateType != 'single' else 1

    cdef long[:, :, ::1] r    = np.ndarray((nThreads, steps, sampleSize), \
                                           dtype = long)
    # cdef long[:, :, ::1] r = np.ndarray((nThreds, steps, sampleSize), dtype = long)

    pbar = tqdm(total = nSamples)
    for sample in prange(nSamples, nogil = True, \
                         schedule = 'static', num_threads = nThreads):

        tid = threadid()
        r[tid] = (<Model> models_[tid].ptr).sampleNodes(steps)
        # r[sample] = (<Model> models_[sample].ptr).sampleNodes(steps)
        # perform n steps
        for step in range(steps):
            # (<Model> models_[sample].ptr)._updateState(\
                                                    # r[(sample + 1) * (step + 1) - 1]
                                                        # )
            (<Model> models_[tid].ptr)._updateState(\
                                                    r[tid, step]
                                                        )
        with gil:
            idx = encodeState((<Model> models_[tid].ptr)._states)
            # idx = encodeState((<Model> models_[sample].ptr)._states)
            snapshots[idx] += 1/Z
            pbar.update(1)
    print('done')
    # pbar = tqdm(total = nSamples)
    # model.reset() # start from random
    # for i in range(N):
    #     if i % step == 0:
    #         idx             = encodeState(model._states)
    #         snapshots[idx] += 1 / Z
    #         pbar.update(1)
    #     # model._updateState(r[i])
    #     model._updateState(r[i])
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
               Model model, dict snapshots,
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
        double[::1] copyNudge = model.nudges.copy()
        bint reset            = True
        # loop stuff
        # extract startstates
        # list comprehension is slower than true loops cython
        long[:, ::1] s = np.array([decodeState(i, model._nNodes) for i in tqdm(snapshots)])
        int states     = len(snapshots)

        # CANT do this inline which sucks either assign it below with gill or move this to proper c/c++
        # loop parameters
        int repeat, delta, node, statei, half = deltas // 2, state
        vector[int] kdxs        = list(snapshots.keys()) # extra mapping idx
        dict conditional = {}
        # unordered_map[int, double *] conditional
        long[::1] startState
        int jdx

        # long[  :,       ::1] r       = model.sampleNodes( states * (deltas) * repeats)
        # list m = []

        int nNodes = model._nNodes, nStates = model._nStates
        long[::1] agentStates = model.agentStates
        str nudgeType = model._nudgeType

        unordered_map[int, int] idxer = {state : idx for idx, state in enumerate(agentStates)}

        list modelsPy = []
        vector[PyObjectHolder] models_
        Model tmp

    # setup thread count
    nThreads = mp.cpu_count() if nThreads == -1 else nThreads
    # setup output storage

    # threadsafe model access; can be reduces to n_threads
    for state in range(nThreads):
        tmp = copy.deepcopy(model)
        tmp.seed += state # enforce different seeds
        modelsPy.append(tmp)
        models_.push_back(PyObjectHolder(<PyObject *> tmp))


    cdef int sampleSize = model.nNodes if model.updateType != 'single' else 1

    # pre-define rng matrix
    # cdef long[:, ::1] r = model.sampleNodes(states * deltas * repeats)

    # bin matrix
    cdef double[:, :, :, ::1] out = np.zeros((nThreads     , deltas, \
                                              model._nNodes, model._nStates))
    cdef long[:, :, ::1] r = np.ndarray((nThreads, deltas * repeats, sampleSize), dtype = long)


    pbar = tqdm(total = states) # init  progbar
    cdef int nTrial = deltas * repeats
    cdef int tid  # init thread id
    print('starting runs')
    with nogil, parallel(num_threads = nThreads):
        for state in prange(states, schedule = 'static'):
            tid = threadid()
            r[tid] = (<Model>models_[tid].ptr).sampleNodes(nTrial)
            for repeat in range(repeats):
                # only copy values
                for node in range(nNodes):
                    # kinda uggly syntax
                    (<Model>models_[tid].ptr)._states[node] = s[state][node]
                    (<Model>models_[tid].ptr)._nudges[node] = copyNudge[node]
                # sample for N times
                for delta in range(deltas):
                    # bin data
                    for node in range(nNodes):
                        out[tid, delta, node, idxer[(<Model>models_[tid].ptr)._states[node]]] += 1 / Z
                    # update
                    jdx = (delta + 1) * (repeat + 1)#  * (state + 1)
                    (<Model>models_[tid].ptr)._updateState(r[tid, jdx - 1])

                    if nudgeType == 'pulse' or \
                    nudgeType    == 'constant' and delta >= half:
                        # (<Model>models[n])._nudges[:] =
                        (<Model>models_[tid].ptr)._nudges[:] = 0
            # TODO: replace this with a concurrent unordered_map
            with gil:
                pbar.update(1)
                conditional[kdxs[state]] = out.base[tid].copy()# [state, 0, 0, 0]
    # cdef unordered_map[int, double *].iterator start = conditional.begin()
    # cdef unordered_map[int, double *].iterator end   = conditional.end()
    # cdef int length = (deltas + 1) * nNodes * nStates
    # cdef np.ndarray buffer = np.zeros(length)
    # cdef _tmp = {}
    # while start != end:
    #     idx = deref(start).first
    #     for state in range(length):
    #         buffer[state] = deref(start).second[state]
    #     _tmp[idx] = buffer.reshape((deltas + 1, nNodes, nStates)).copy()
    #     prec(start)

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
    # @cython.boundscheck(False) # compiler directive
    # @cython.wraparound(False) # compiler directive
    # @cython.nonecheck(False)
    # @cython.cdivision(True)
    # cpdef dict monteCarlo(\
    #                Model model, dict snapshots,
    #                int deltas = 10,  int repeats = 11,
    #                ):
    #
    #     cdef float past = time.process_time()
    #      # store nudges already there
    #     cdef list models = []
    #     cdef dict params
    #     import copy
    #     params = dict(\
    #                 model      = model,\
    #                 # graph      = model.graph,\
    #                 # nudges     = model.nudges.base.copy(),\
    #                 temp       = model.t,\
    #                 repeats    = repeats,\
    #                 deltas     = deltas,\
    #                 )
    #     from functools import partial
    #     f = partial(worker, **params)
    #     print(f)
    #     cdef np.ndarray s = np.array([q for q in snapshots])
    #     cdef int n = len(s) // (mp.cpu_count() - 1)
    #     if n == 0:
    #         n = 1
    #     cdef list states  = [s[i : i + n] for i in range(0, len(s), n)]
    #     cdef dict conditional = {}
    #     # with mp.Pool(2) as p:
    #     with mp.Pool(mp.cpu_count() - 1) as p:
    #         for res in p.imap(f, tqdm(states)):
    #             for k, v in res.items():
    #                 conditional[k] = v
    #         # conditional = {kdx : res for kdx, res in zip(snapshots, p.map(f, tqdm(models)))}
    #     # print(conditional)
    #     print(f"Delta = {time.process_time() - past}")
    #     return conditional
    #
    #
    #     # object graph,\
    #     # np.ndarray nudges,\
    # @cython.boundscheck(False) # compiler directive
    # @cython.wraparound(False) # compiler directive
    # @cython.nonecheck(False)
    # @cython.cdivision(True)
    # @cython.initializedcheck(False)
    # cpdef dict worker(\
    #                 np.ndarray idx,\
    #                 Model model,\
    #                 double temp,\
    #                 int repeats, \
    #                 int deltas,\
    #                   ):
    #     # setup the worker
    #     # cdef Ising model = copy.deepcopy(Ising(graph, temperature = temp, updateType = 'single'))
    #     # model.nudges     = nudges.copy()
    #     # cdef Model model = copy.deepcopy(m)
    #     # cdef Ising model = copy.deepcopy(model)
    #     # model.nudges = nudges.copy()
    #     print(id(model))
    #     cdef dict conditional = {}
    #     # print(model.seed)
    #     cdef int states            = idx.size
    #     # decode the states
    #     cdef int nNodes            = model.nNodes
    #     cdef int nStates           = model.nStates
    #     cdef str nudgeType         = model.nudgeType
    #     cdef double[::1] copyNudge = model.nudges.base.copy()
    #
    #     cdef long[:, ::1] s = np.asarray([decodeState(i, nNodes) for i in idx])
    #     cdef long[:, ::1] r = model.sampleNodes( states * (deltas + 1) * repeats)
    #     # mape state to index
    #     cdef unordered_map[int, int] idxer = {i : j for j, i in enumerate(model.agentStates)}
    #     cdef double[:, :, :, ::1] out = np.zeros((states, deltas + 1, nNodes, nStates))
    #     cdef int half = deltas // 2
    #     cdef state, repeat, node, jdx
    #     cdef double Z = <double> repeats
    #     print(id(model), id(model.states.base), mp.current_process().name, id(model.nudges.base))
    #     cdef bint reset
    #     for state in range(states):
    #         # with gil:
    #         for repeat in range(repeats):
    #             # reset the buffers to the start state
    #             # model._states[:] = s[state]
    #             # model._nudges[:] = copyNudge
    #             for node in range(nNodes):
    #                 model._states[node] = s[state][node]
    #                 model._nudges[node] = copyNudge[node]
    #             # reset simulation
    #             reset   = True
    #             # sample for N times
    #             for delta in range(deltas + 1):
    #                 # bin data
    #                 for node in range(nNodes):
    #                     out[state, delta, node, idxer[model._states[node]]] += 1 / Z
    #                 # update
    #                 jdx  = (delta + 1) * (repeat + 1)  * (state + 1) - 1
    #                 # (<Model>models[n])._updateState(r[jdx])
    #                 # model._updateState(model.sampleNodes(1)[0])
    #                 model._updateState(r[jdx])
    #                 # turn-off the nudges
    #                 if reset:
    #                     # check for type of nudge
    #                     if nudgeType == 'pulse' or \
    #                     nudgeType    == 'constant' and delta >= half:
    #                         for node in range(nNodes):
    #                             model._nudges[node] = 0
    #                         reset            = False
    #         # pbar.update(1)
    #         conditional[idx[state]] = out.base[state]
    #     return conditional
    #
    # #


# # belongs to worker
# @cython.boundscheck(False) # compiler directive
# @cython.wraparound(False) # compiler directive
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cpdef dict monteCarlo(\
#                Model model, dict snapshots,
#                int deltas = 10,  int repeats = 11,
#                ):
#
#     cdef float past = time.process_time()
#      # store nudges already there
#     cdef list models = []
#     cdef dict params
#     import copy
#     for startidx, val in snapshots.items():
#         params = dict(\
#                     model      = model,\
#                     repeats    = repeats,\
#                     deltas     = deltas,\
#                     idx        = startidx,\
#                     startState = np.asarray(decodeState(startidx, model._nNodes)),\
#                     )
#
#         models.append(Worker(**params))
#     # cdef np.ndarray s = np.array([decodeState(q, model._nNodes) for q in snapshots], ndmin = 2)
#     cdef dict conditional = {}
#     # with mp.Pool(2) as p:
#     with mp.Pool(3) as p:
#         for kdx, res in zip(snapshots, p.apply_async(f, tqdm(models)):
#             conditional[kdx] = res
#         # conditional = {kdx : res for kdx, res in zip(snapshots, p.map(f, tqdm(models)))}
#     print(conditional)
#     print(f"Delta = {time.process_time() - past}")
#     return conditional


cpdef np.ndarray f(Worker x):
    # print('id', id(x))
    return x.parallWrap()


@cython.auto_pickle(True)
cdef class Worker:
    """
    This class was used to wrap the c classes for use with the multiprocessing toolbox.
    However, the performance decreased a lot. I think this may be a combination of the nogil
    sections. Regardless the 'single' threaded stuff above is plenty fast for me.
    Future me should look into dealing with the gil  and wrapping everything in  c arrays
    """
    cdef int deltas
    cdef int idx
    cdef int repeats
    cdef np.ndarray startState
    cdef Model model
    # cdef dict __dict__
    def __init__(self, *args, **kwargs):
        # for k, v in kwargs.items():
        #     setattr(self, k, v)

        self.deltas     = kwargs['deltas']
        self.model      = kwargs['model']
        self.repeats    = kwargs['repeats']
        self.startState = kwargs['startState']
        self.idx        = kwargs['idx']

    cdef np.ndarray parallWrap(self):
        cdef long[::1] startState = self.startState
        # start unpacking
        cdef int deltas           = self.deltas
        cdef int repeats          = self.repeats
        # cdef long[::1] startState = startState
        cdef Model model          = self.model
        # pre-declaration
        cdef double[::1] out = np.zeros((deltas + 1) * model._nNodes * model._nStates)
        cdef double Z              = <double> repeats
        cdef double[:] copyNudge   = model._nudges.copy()
        cdef bint reset            = True
        # loop stuff
        cdef long[:, ::1] r
        cdef int k, delta, node, statei, counter, half = deltas // 2
        # pbar = tqdm(total = repeats)
        for k in range(repeats):
            for node in range(model._nNodes):
                model._states[node] = startState[node]
                model._nudges[node] = copyNudge[node]
            # reset simulation
            reset   = True
            counter = 0
            r       = model.sampleNodes(repeats * (deltas + 1))
            for delta in range(deltas + 1):
                # bin data
                for node in range(model._nNodes):
                    for statei in range(model._nStates):
                        if model._states[node] == model.agentStates[statei]:
                            out[counter] += 1 / Z
                        counter += 1
                # update
                model._updateState(r[counter])

                # turn-off
                if reset:
                    if model._nudgeType == 'pulse' or \
                    model._nudgeType    == 'constant' and delta >= half:
                        model._nudges[:] = 0
                        reset            = False
            # pbar.update(1)
        # pbar.close()
        return out.base.reshape((deltas + 1, model._nNodes, model._nStates))
