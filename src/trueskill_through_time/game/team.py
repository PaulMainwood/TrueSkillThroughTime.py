from ..core.gaussian import Ninf, N00
from math import inf
from ..config import *
from ..core.math_utils import *
class Team:
    def __init__(self, items, output):
        self.items = items
        self.output = output

class Item:
    def __init__(self, name, likelihood):
        self.name = name
        self.likelihood = likelihood

class Agent(object):
    def __init__(self, player, message, last_time):
        self.player = player
        self.message = message
        self.last_time = last_time
    
    def receive(self, elapsed):
        if self.message != Ninf:
            res = self.message.forget(self.player.gamma, elapsed) 
        else:
            res = self.player.prior
        return res

def clean(agents,last_time=False):
    for a in agents:
        agents[a].message = Ninf
        if last_time:

            agents[a].last_time = -inf

class team_variable(object):
    def __init__(self, prior=Ninf, likelihood_lose=Ninf, likelihood_win=Ninf, likelihood_draw=Ninf):
        self.prior = prior
        self.likelihood_lose = likelihood_lose
        self.likelihood_win = likelihood_win
        self.likelihood_draw = likelihood_draw
        
    @property
    def p(self):
        return self.prior*self.likelihood_lose*self.likelihood_win*self.likelihood_draw
    @property
    def posterior_win(self):
        return self.prior*self.likelihood_lose*self.likelihood_draw
    @property
    def posterior_lose(self):
        return self.prior*self.likelihood_win*self.likelihood_draw
    @property
    def likelihood(self):
        return self.likelihood_win*self.likelihood_lose*self.likelihood_draw

def performance(team, weights):
    res = N00
    for player, w in zip(team, weights):
        res += player.performance() * w
    return res

class draw_messages(object):
    def __init__(self,prior = Ninf, prior_team = Ninf, likelihood_lose = Ninf, likelihood_win = Ninf):
        self.prior = prior
        self.prior_team = prior_team
        self.likelihood_lose = likelihood_lose
        self.likelihood_win = likelihood_win
    
    @property
    def p(self):
        return self.prior_team*self.likelihood_lose*self.likelihood_win
    
    @property
    def posterior_win(self):
        return self.prior_team*self.likelihood_lose
    
    @property
    def posterior_lose(self):
        return self.prior_team*self.likelihood_win
    
    @property
    def likelihood(self):
        return self.likelihood_win*self.likelihood_lose

class diff_messages(object):
    def __init__(self, prior=Ninf, likelihood=Ninf):
        self.prior = prior
        self.likelihood = likelihood
    @property
    def p(self):
        return self.prior*self.likelihood