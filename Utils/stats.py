import numpy as np
from Utils.plotting import fit
'''Dump file of statistical related functions'''

def aucs(data, func, params = {},\
        bounds = (0, np.inf), **kwargs):
    from scipy.integrate import quad
    coeffs = fit(data, func, params = params, **kwargs)[0]
    auc = np.zeros((data.shape[0]), dtype = float)
    for node, c in enumerate(coeffs):
        tmp = lambda x: func(x, *c)
        auc[node] = quad(tmp, *bounds)[0]
    return auc

from statsmodels.stats.proportion import proportion_confint

def pci(ranks, alpha = .05):
    """
    Compute confidence interval on binomial successes
    with willson correction
    """
    from collections import Counter
    hist = Counter(ranks)

#     pvals = {}
    from scipy.stats import norm, multinomial, chi2_contingency
#     z = norm.ppf(1 - alpha / 2)
#     n = len(hist)
    n = len(ranks)

    hist = dict(sorted(hist.items(), key = lambda x : x[1])[::-1])

    candidates = [i for i in hist.keys()]
    success = [i for i in hist.values()]

#     print(success)
    for ni in range(len(hist)):
        p = multinomial.pmf(success[:ni + 1], n = n, p = np.ones(ni + 1) * 1/(ni + 1))
#         chi2, p, dof, expected = chi2_contingency(t,  correction = 0)
#         print(chi2, p, dof, expected)
        driverset = set(candidates[:ni + 1])
        if p >= alpha:
#             print(driverset, hist)
            break

    return driverset

def driverNodeEstimateSEM(ranks, alpha = .01):
    """
    Use all the trials for all the nodes to determine the driver node
    :ranks: should be node x trials
    """

    # get number of standard deviations
    from scipy.stats import norm, sem
    zdx = norm.ppf(1 - alpha / 2)
    nodes, trials = ranks.shape

    means = ranks.mean(1)
    sems  = sem(ranks, axis = 1) * zdx

    driver = means.argmax()

    candidates = set([driver])
    for i in range(nodes):
        if i != driver:
            if means[i] + sems[i] >= means[driver] - sems[driver]:
                candidates.add(i)
    return candidates




def driverNodeEstimate(ranks, alpha = .01):
    from collections import Counter
    from statsmodels.stats.proportion import multinomial_proportions_confint as mpci

    hist = Counter(ranks)
    hist = sorted(hist.items(), key = lambda x : x[1])[::-1]
    candidates = [i[0] for i in hist]
#     print(candidates, hist)
    if len(candidates) > 1:
        intervals = mpci([i[1] for i in hist], alpha = alpha)
        driverSet = set([candidates[0]])
        # print(intervals)
        for idx, (lower, upper) in enumerate(intervals[1:]):
            if upper > intervals[0, 0]:
                driverSet.add(candidates[idx + 1])
        return driverSet
    else:
        return set(candidates)

def determineDrivers(drivers, alpha = 0.05):
    """
    Iterative approach in determining  the driver-nodes

    The test involves iteratively testing whether there exist i-driver nodes in the rank
    using the success rate of a binomial distribution

    Input:
        :drivers: vector of nTrials long
        :alpha: rejection alpha
    """
    from scipy.stats import binom
    from collections import Counter # histogram symbolic
    n = len(drivers) #.shape[0]
    # bin the data
    # get distribution over rank 1; start from highest
    counts = sorted(Counter(drivers).items(), key = lambda x : x[1])[::-1]

    nDrivers = len(counts)
    # print(counts, drivers)
    # iteratively test how many driver-nodes there are
    # print(nDrivers)
    if nDrivers == 1:
        return 1, counts
    else:
        for nDriver in range(1, nDrivers):

            driverSet = counts[:nDriver + 1]
            # H_0 -> are there nDriver + 1 driver nodes?
            p = 1  # assume the random choice
            success = sum([count for node, count in driverSet])
            pval = binom.pmf(success, n, p)

            if pval  > alpha:
                break
            # return pval, driverSet
    # print(driverSet)
    return pval, driverSet


def hellingerDistance(p1, p2):
    '''
    input:
        :p1: probability control
        :p2: probability of nudge conditions
    Note: it assumes the last axis contains the probability distributions to be compared
    returns:
        hellinger distance between p1, p2
    '''
    return np.linalg.norm( p1 - p2, ord = 2, axis = -1) / np.sqrt(2)

def KL(p1, p2, exclude = []):
    # p2[p2 == 0] = 1 # circumvent p = 0
    # p1[p1==0] = 1
    kl = np.nansum(p1 * (np.log(p1 / p2)), axis = -1)
    kl[np.isfinite(kl) == False] = 0 # remove x = 0
    if exclude:
        kl = np.array([i for idx, i in enumerate(kl) if idx not in exclude])
    return kl

def panzeriTrevesCorrection(px, cpx, repeats):
    """
    Panzeri-Treves sampling bias correction
    Input:
        :px: ndarray node distribution
        :cpx: dict conditional distribution
        :repeats: constant
    Returns:
        :bias:
    """
    rs  = np.zeros(px.shape[:-1])
    for key, value in cpx.items():
        for zdx, deltaInfo in enumerate(value):
            for jdx, nodeInfo in enumerate(deltaInfo):
                rs[zdx, jdx] += pt_bayescount(nodeInfo, repeats) - 1
    Rs = np.array([[pt_bayescount(j, repeats) - 1 for j in i]\
                 for i in px])

    return (rs - Rs) / (2 * repeats * np.log(2))
def pt_bayescount(Pr, Nt):
    # all credit of this function goes to panzeri-treves
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
def KL2(p1, p2):
    from scipy.special import kl_div
    return np.nansum(kl_div(p1, p2), axis = -1)

def JS(p1, p2):
    """
    Jenson shannon divergence
    """
    return KL(p1, p2) / 2 + KL(p2, p1)



# -*- coding: utf-8 -*-

u"""
Beta regression for modeling rates and proportions.
References
----------
Grün, Bettina, Ioannis Kosmidis, and Achim Zeileis. Extended beta regression
in R: Shaken, stirred, mixed, and partitioned. No. 2011-22. Working Papers in
Economics and Statistics, 2011.
Smithson, Michael, and Jay Verkuilen. "A better lemon squeezer?
Maximum-likelihood regression with beta-distributed dependent variables."
Psychological methods 11.1 (2006): 54.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.special import gammaln as lgamma

from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod.families import Binomial

# this is only need while #2024 is open.
class Logit(sm.families.links.Logit):

    """Logit tranform that won't overflow with large numbers."""

    def inverse(self, z):
        return 1 / (1. + np.exp(-z))

_init_example = """
    Beta regression with default of logit-link for exog and log-link
    for precision.
    >>> mod = Beta(endog, exog)
    >>> rslt = mod.fit()
    >>> print rslt.summary()
    We can also specify a formula and a specific structure and use the
    identity-link for phi.
    >>> from sm.families.links import identity
    >>> Z = patsy.dmatrix('~ temp', dat, return_type='dataframe')
    >>> mod = Beta.from_formula('iyield ~ C(batch, Treatment(10)) + temp',
    ...                         dat, Z=Z, link_phi=identity())
    In the case of proportion-data, we may think that the precision depends on
    the number of measurements. E.g for sequence data, on the number of
    sequence reads covering a site:
    >>> Z = patsy.dmatrix('~ coverage', df)
    >>> mod = Beta.from_formula('methylation ~ disease + age + gender + coverage', df, Z)
    >>> rslt = mod.fit()
"""

class Beta(GenericLikelihoodModel):

    """Beta Regression.
    This implementation uses `phi` as a precision parameter equal to
    `a + b` from the Beta parameters.
    """

    def __init__(self, endog, exog, Z=None, link=Logit(),
            link_phi=sm.families.links.Log(), **kwds):
        """
        Parameters
        ----------
        endog : array-like
            1d array of endogenous values (i.e. responses, outcomes,
            dependent variables, or 'Y' values).
        exog : array-like
            2d array of exogeneous values (i.e. covariates, predictors,
            independent variables, regressors, or 'X' values). A nobs x k
            array where `nobs` is the number of observations and `k` is
            the number of regressors. An intercept is not included by
            default and should be added by the user. See
            `statsmodels.tools.add_constant`.
        Z : array-like
            2d array of variables for the precision phi.
        link : link
            Any link in sm.families.links for `exog`
        link_phi : link
            Any link in sm.families.links for `Z`
        Examples
        --------
        {example}
        See Also
        --------
        :ref:`links`
        """.format(example=_init_example)
        assert np.all((0 < endog) & (endog < 1))
        if Z is None:
            extra_names = ['phi']
            Z = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['precision-%s' % zc for zc in \
                        (Z.columns if hasattr(Z, 'columns') else range(1, Z.shape[1] + 1))]
        kwds['extra_params_names'] = extra_names

        super(Beta, self).__init__(endog, exog, **kwds)
        self.link = link
        self.link_phi = link_phi

        self.Z = Z
        assert len(self.Z) == len(self.endog)

    def nloglikeobs(self, params):
        """
        Negative log-likelihood.
        Parameters
        ----------
        params : np.ndarray
            Parameter estimates
        """
        return -self._ll_br(self.endog, self.exog, self.Z, params)

    def fit(self, start_params=None, maxiter=100000, maxfun=5000, disp=False,
            method='bfgs', **kwds):
        """
        Fit the model.
        Parameters
        ----------
        start_params : array-like
            A vector of starting values for the regression
            coefficients.  If None, a default is chosen.
        maxiter : integer
            The maximum number of iterations
        disp : bool
            Show convergence stats.
        method : str
            The optimization method to use.
        """

        if start_params is None:
            start_params = sm.GLM(self.endog, self.exog, family=Binomial()
                                 ).fit(disp=False).params
            start_params = np.append(start_params, [0.5] * self.Z.shape[1])

        return super(Beta, self).fit(start_params=start_params,
                                        maxiter=maxiter, maxfun=maxfun,
                                        method=method, disp=disp, **kwds)

    def _ll_br(self, y, X, Z, params):
        nz = self.Z.shape[1]

        Xparams = params[:-nz]
        Zparams = params[-nz:]

        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_phi.inverse(np.dot(Z, Zparams))
        # TODO: derive a and b and constrain to > 0?

        if np.any(phi <= np.finfo(float).eps): return np.array(-np.inf)

        ll = lgamma(phi) - lgamma(mu * phi) - lgamma((1 - mu) * phi) \
                + (mu * phi - 1) * np.log(y) + (((1 - mu) * phi) - 1) \
                * np.log(1 - y)

        return ll
