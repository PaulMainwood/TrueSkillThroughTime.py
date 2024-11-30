# core/gaussian.py
import math
from ..config import MU, SIGMA
from .math_utils import inf, mu_sigma

class Gaussian(object):
    """
    The `Gaussian` class is used to define the prior beliefs of the agents' skills
    
    Attributes
    ----------
    mu : float
        the mean of the `Gaussian` distribution.
    sigma :
        the standar deviation of the `Gaussian` distribution.
        
    """
    def __init__(self,mu=MU, sigma=SIGMA):
        if sigma >= 0.0:
            self.mu, self.sigma = mu, sigma
        else:
            raise ValueError(" sigma should be greater than 0 ")
    
    @property
    def tau(self):
        if self.sigma > 0.0:
            return self.mu * (self.sigma**-2)
        else:
            return inf
        
    @property
    def pi(self):
        if self.sigma > 0.0:
            return self.sigma**-2
        else:
            return inf
    
    def __iter__(self):
        return iter((self.mu, self.sigma))
    def __repr__(self):
        return 'N(mu={:.3f}, sigma={:.3f})'.format(self.mu, self.sigma)
    def __add__(self, M):
        return Gaussian(self.mu + M.mu, math.sqrt(self.sigma**2 + M.sigma**2))
    def __sub__(self, M):
        return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 + M.sigma**2))
    def __mul__(self, M):
        if type(M) == float:
            if M == inf: 
                return Ninf
            else:
                return Gaussian(M*self.mu, abs(M)*self.sigma)
        else:
            if self.sigma == 0.0 or M.sigma == 0.0:
                mu = self.mu/((self.sigma**2/M.sigma**2) + 1) if self.sigma == 0.0 else M.mu/((M.sigma**2/self.sigma**2) + 1)
                sigma = 0.0
            else:
                _tau, _pi = self.tau + M.tau, self.pi + M.pi
                mu, sigma = mu_sigma(_tau, _pi)
            return Gaussian(mu, sigma)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, M):
        _tau = self.tau - M.tau; _pi = self.pi - M.pi
        mu, sigma = mu_sigma(_tau, _pi)
        return Gaussian(mu, sigma)
    def forget(self,gamma,t):
        return Gaussian(self.mu, math.sqrt(self.sigma**2 + t*gamma**2))
    def delta(self, M):
        return abs(self.mu - M.mu) , abs(self.sigma - M.sigma) 
    def exclude(self, M):
        return Gaussian(self.mu - M.mu, math.sqrt(self.sigma**2 - M.sigma**2) )
    def isapprox(self, M, tol=1e-4):
        return (abs(self.mu - M.mu) < tol) and (abs(self.sigma - M.sigma) < tol)
     
N01 = Gaussian(0,1)
N00 = Gaussian(0,0)
Ninf = Gaussian(0,inf)
Nms = Gaussian(MU, SIGMA)
