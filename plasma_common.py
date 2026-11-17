"""Shared EXACT stationary fluid--Poisson/Sagdeev calculations.

Normalisation and EOS follow the five original MATLAB scripts. This module
contains no proprietary calls or external datasets. Maintained as a small
shared numerical library so every `python figN.py` invocation works from
any working directory when the files are kept together.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

@dataclass(frozen=True)
class EOS:
    sigma: float = 0.2
    p1: float = 1.0
    p2: float = -6.0
    p3: float = 35.21186973360886

    @property
    def lam(self) -> float:
        return np.sqrt(1+self.sigma*self.p1)

    @property
    def p2c(self) -> float:
        return -self.p1 - 1/self.sigma

    @property
    def p3c(self) -> float:
        return self.lam**2/self.sigma

    @property
    def beta_c(self) -> float:
        return 3*(self.sigma*self.p3-self.lam**2)/(2*self.lam)

def p_diff(n, p2, p3, p1=1.0):
    y=np.asarray(n)-1
    return p1*y+p2*y**2+p3*y**3

def p_prime(n,p2,p3,p1=1.0):
    y=np.asarray(n)-1
    return p1+2*p2*y+3*p3*y**2

def enthalpy(n,p2,p3,p1=1.0):
    """Analytic integral int_1^n p'(s)/s ds. Use log1p near 1."""
    y=np.asarray(n)-1
    return ((p1-2*p2+3*p3)*np.log1p(y)
            +(2*p2-6*p3)*y+1.5*p3*y*(np.asarray(n)+1))

def bernoulli(n,M,eos:EOS,p2=None,p3=None):
    p2=eos.p2 if p2 is None else p2
    p3=eos.p3 if p3 is None else p3
    y=np.asarray(n)-1
    return .5*M*M*y*(np.asarray(n)+1)/np.asarray(n)**2 - eos.sigma*enthalpy(n,p2,p3,eos.p1)

def bernoulli_prime(n,M,eos:EOS,p2=None,p3=None):
    p2=eos.p2 if p2 is None else p2
    p3=eos.p3 if p3 is None else p3
    n=np.asarray(n)
    return M*M/n**3-eos.sigma*p_prime(n,p2,p3,eos.p1)/n

def sagdeev(n,M,eos:EOS,p2=None,p3=None):
    """S(phi(n);M) = M^2(1-1/n)-sigma[p(n)-p(1)]-expm1(phi)."""
    p2=eos.p2 if p2 is None else p2
    p3=eos.p3 if p3 is None else p3
    n=np.asarray(n)
    return M*M*(n-1)/n-eos.sigma*p_diff(n,p2,p3,eos.p1)-np.expm1(bernoulli(n,M,eos,p2,p3))

def first_turning_point(eos:EOS, M, p2=None,p3=None, polarity=1,
                        r_min=.003,r_max=.5,n_grid=1700, tol=1e-8):
    """Port of the MATLAB *first* root and orbit admissibility checks.

    Returns (signed potential amplitude, turning density). NaN means no
    validated root in the search interval, NOT a proof of nonexistence.
    """
    p2=eos.p2 if p2 is None else p2
    p3=eos.p3 if p3 is None else p3
    rr=np.linspace(r_min,r_max,n_grid)
    nn=1+polarity*rr
    if np.any(nn<=0):
        raise ValueError('Density scan reached a nonpositive value')
    ss=sagdeev(nn,M,eos,p2,p3)
    bp=bernoulli_prime(nn,M,eos,p2,p3)
    if not np.isfinite(ss[0]) or ss[0]>=0 or bp[0]<=0:
        return np.nan,np.nan
    indices=np.where(np.isfinite(ss[:-1]) & np.isfinite(ss[1:]) & (ss[:-1]<0) & (ss[1:]>=0))[0]
    if not len(indices):
        return np.nan,np.nan
    j=int(indices[0])
    if np.any(~np.isfinite(bp[:j+2])) or np.any(bp[:j+2]<=1e-11):
        return np.nan,np.nan
    try:
        root=brentq(lambda n: float(sagdeev(n,M,eos,p2,p3)),
                    min(nn[j],nn[j+1]),max(nn[j],nn[j+1]),xtol=1e-13)
    except ValueError:
        return np.nan,np.nan
    ns=np.linspace(1+polarity*r_min,root,400)
    if np.any(sagdeev(ns,M,eos,p2,p3)>tol) or np.any(bernoulli_prime(ns,M,eos,p2,p3)<=1e-11):
        return np.nan,np.nan
    A=float(bernoulli(root,M,eos,p2,p3))
    if polarity*A<=0:
        return np.nan,np.nan
    return A,root
