import trueskill_through_time.config
from trueskill_through_time.game.team import Agent
from trueskill_through_time.temporal.batch import Batch
from trueskill_through_time.core.player import Player
import trueskill_through_time.utils.sorting as sorting
from trueskill_through_time.core.gaussian import *
from ..config import *

class History(object):
    def __init__(self,composition, results=[], times=[], priors=dict(), mu=MU, sigma=SIGMA, beta=BETA, gamma=GAMMA, p_draw=P_DRAW, weights=[]):
        if (len(results) > 0) and (len(composition) != len(results)): raise ValueError("len(composition) != len(results)")
        if (len(times) > 0) and (len(composition) != len(times)): raise ValueError(" len(times) error ")
        if (len(weights) > 0) and (len(composition) != len(weights)): raise ValueError("(length(weights) > 0) & (length(composition) != length(weights))")
            
        self.size = len(composition)
        self.batches = []
        self.agents = dict([ (a, Agent(priors[a] if a in priors else Player(Gaussian(mu, sigma), beta, gamma), Ninf, -inf)) for a in set( [a for teams in composition for team in teams for a in team] ) ])
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.p_draw = p_draw
        self.time = len(times)>0
        self.trueskill(composition,results,times, weights)
        
    def __repr__(self):
        return "History(Events={}, Batches={}, Agents={})".format(self.size,len(self.batches),len(self.agents))
    def __len__(self):
        return self.size
    def trueskill(self, composition, results, times, weights):
        o = sorting.sortperm(times) if len(times)>0 else [i for i in range(len(composition))]
        i = 0
        while i < len(self):
            j, t = i+1, i+1 if len(times) == 0 else times[o[i]]
            while (len(times)>0) and (j < len(self)) and (times[o[j]] == t): j += 1
            if len(results) > 0:
                b = Batch([composition[k] for k in o[i:j]],[results[k] for k in o[i:j]], t, self.agents, self.p_draw, weights if not weights else [weights[k] for k in o[i:j]])
            else:
                b = Batch([composition[k] for k in o[i:j]],[], t, self.agents, self.p_draw, weights if not weights else [weights[k] for k in o[i:j]])
            self.batches.append(b)
            for a in b.skills:
                self.agents[a].last_time = t if self.time else inf
                self.agents[a].message = b.forward_prior_out(a)
            i = j
    def iteration(self):
        step = (0., 0.)
        clean(self.agents)
        for j in reversed(range(len(self.batches)-1)):
            for a in self.batches[j+1].skills:
                self.agents[a].message = self.batches[j+1].backward_prior_out(a)
            old = self.batches[j].posteriors().copy()
            self.batches[j].new_backward_info()
            step = max_tuple(step, dict_diff(old, self.batches[j].posteriors()))
        clean(self.agents)
        for j in range(1,len(self.batches)):
            for a in self.batches[j-1].skills:
                self.agents[a].message = self.batches[j-1].forward_prior_out(a)
            old = self.batches[j].posteriors().copy()
            self.batches[j].new_forward_info()
            step = max_tuple(step, dict_diff(old, self.batches[j].posteriors()))
    
        if len(self.batches)==1:
            old = self.batches[0].posteriors().copy()
            self.batches[0].convergence()
            step = max_tuple(step, dict_diff(old, self.batches[0].posteriors()))
        
        return step
    def convergence(self, epsilon = EPSILON, iterations = ITERATIONS, verbose=True):
        step = (inf, inf); i = 0
        while gr_tuple(step, epsilon) and (i < iterations):
            if verbose: print("Iteration = ", i, end=" ")
            step = self.iteration()
            i += 1
            if verbose: print(", step = ", step)
        if verbose: print("End")
        return step, i
    def learning_curves(self):
        res = dict()
        for b in self.batches:
            for a in b.skills:
                t_p = (b.time, b.posterior(a))
                if a in res:
                    res[a].append(t_p)
                else:
                    res[a] = [t_p]
        return res
    def log_evidence(self):
        return sum([math.log(event.evidence) for b in self.batches for event in b.events])