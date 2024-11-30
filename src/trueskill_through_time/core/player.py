from ..config import *
from .gaussian import *
from math import sqrt, inf

class Player:
    def __init__(self, prior=Gaussian(MU, SIGMA), beta=BETA, gamma=GAMMA, prior_draw=None):
        self.prior = prior
        self.beta = beta
        self.gamma = gamma
        self.prior_draw = prior_draw or Gaussian(0, inf)
    
    def performance(self):
        return Gaussian(
            self.prior.mu,
            sqrt(self.prior.sigma**2 + self.beta**2)
        )
        
    def __repr__(self):
        return f'Player(Gaussian(mu={self.prior.mu:.3f}, sigma={self.prior.sigma:.3f}), beta={self.beta:.3f}, gamma={self.gamma:.3f})'