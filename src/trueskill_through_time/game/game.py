from ..utils.sorting import sortperm
from .team import *
from ..core.math_utils import *
from ..core.gaussian import Gaussian

class Game(object):
    def __init__(self, teams, result = [], p_draw=0.0, weights=[]):
        if len(result) and (len(teams) != len(result)): raise ValueError("len(result) and (len(teams) != len(result))")
        if (0.0 > p_draw) or (1.0 <= p_draw): raise ValueError ("0.0 <= proba < 1.0")
        if (p_draw == 0.0) and (len(result)>0) and (len(set(result))!=len(result)): raise ValueError("(p_draw == 0.0) and (len(result)>0) and (len(set(result))!=len(result))")
        if (len(weights)>0) and (len(teams)!= len(weights)):raise ValueError("(len(weights)>0) & (len(teams)!= len(weights))")
        if (len(weights)>0) and (any([len(team) != len(weight) for (team, weight) in zip(teams, weights)])): ValueError("(len(weights)>0) & exists i (len(teams[i]) != len(weights[i])")
        
        self.teams = teams
        self.result = result
        self.p_draw = p_draw
        if not weights:
            weights = [[1.0 for p in t] for t in teams]
        self.weights = weights
        self.likelihoods = []
        self.evidence = 0.0
        self.compute_likelihoods()
        
    def __len__(self):
        return len(self.teams)
    
    def size(self):
        return [len(team) for team in self.teams]
    
    def performance(self,i):
        return performance(self.teams[i], self.weights[i])    
    
    def partial_evidence(self, d, margin, tie, e):
        mu, sigma = d[e].prior.mu, d[e].prior.sigma
        self.evidence *= cdf(margin[e],mu,sigma)-cdf(-margin[e],mu,sigma) if tie[e] else 1-cdf(margin[e],mu,sigma)
    
    def graphical_model(self):
        g = self 
        r = g.result if len(g.result) > 0 else [i for i in range(len(g.teams)-1,-1,-1)] 
        o = sortperm(r, reverse=True) 
        t = [team_variable(g.performance(o[e]),Ninf, Ninf, Ninf) for e in range(len(g))]
        d = [diff_messages(t[e].prior - t[e+1].prior, Ninf) for e in range(len(g)-1)]
        
        tie = [r[o[e]]==r[o[e+1]] for e in range(len(d))]
        margin = [0.0 if g.p_draw==0.0 else compute_margin(g.p_draw, math.sqrt( sum([a.beta**2 for a in g.teams[o[e]]]) + sum([a.beta**2 for a in g.teams[o[e+1]]]) )) for e in range(len(d))] 
        g.evidence = 1.0
        return o, t, d, tie, margin
    
    def likelihood_analitico(self):
        g = self
        o, t, d, tie, margin = g.graphical_model()
        g.partial_evidence(d, margin, tie, 0)
        d = d[0].prior
        mu_trunc, sigma_trunc =  trunc(d.mu, d.sigma, margin[0], tie[0])
        if d.sigma==sigma_trunc:
            delta_div = d.sigma**2*mu_trunc - sigma_trunc**2*d.mu
            theta_div_pow2 = inf
        else:
            delta_div = (d.sigma**2*mu_trunc - sigma_trunc**2*d.mu)/(d.sigma**2-sigma_trunc**2)
            theta_div_pow2 = (sigma_trunc**2*d.sigma**2)/(d.sigma**2 - sigma_trunc**2)
        res = []
        for i in range(len(t)):
            team = []
            for j in range(len(g.teams[o[i]])):
                mu = 0.0 if d.sigma==sigma_trunc else g.teams[o[i]][j].prior.mu + ( delta_div - d.mu)*(-1)**(i==1)
                sigma_analitico = math.sqrt(theta_div_pow2 + d.sigma**2
                                            - g.teams[o[i]][j].prior.sigma**2)
                team.append(Gaussian(mu,sigma_analitico))
            res.append(team)
        return (res[0],res[1]) if o[0]<o[1] else (res[1],res[0])
    
    def likelihood_teams(self):
        g = self 
        o, t, d, tie, margin = g.graphical_model()
        step = (inf, inf); i = 0 
        while gr_tuple(step,1e-6) and (i < 10):
            step = (0., 0.)
            for e in range(len(d)-1):
                d[e].prior = t[e].posterior_win - t[e+1].posterior_lose
                if (i==0): g.partial_evidence(d, margin, tie, e)
                d[e].likelihood = approx(d[e].prior,margin[e],tie[e])/d[e].prior
                likelihood_lose = t[e].posterior_win - d[e].likelihood
                step = max_tuple(step,t[e+1].likelihood_lose.delta(likelihood_lose))
                t[e+1].likelihood_lose = likelihood_lose
            for e in range(len(d)-1,0,-1):
                d[e].prior = t[e].posterior_win - t[e+1].posterior_lose
                if (i==0) and (e==len(d)-1): g.partial_evidence(d, margin, tie, e)
                d[e].likelihood = approx(d[e].prior,margin[e],tie[e])/d[e].prior
                likelihood_win = t[e+1].posterior_lose + d[e].likelihood
                step = max_tuple(step,t[e].likelihood_win.delta(likelihood_win))
                t[e].likelihood_win = likelihood_win
            i += 1
        if len(d)==1:
            g.partial_evidence(d, margin, tie, 0)
            d[0].prior = t[0].posterior_win - t[1].posterior_lose
            d[0].likelihood = approx(d[0].prior,margin[0],tie[0])/d[0].prior
        t[0].likelihood_win = t[1].posterior_lose + d[0].likelihood
        t[-1].likelihood_lose = t[-2].posterior_win - d[-1].likelihood
        return [ t[o[e]].likelihood for e in range(len(t)) ] 
    
    def compute_likelihoods(self):
        if len(self.teams)>2 or len([w for t in self.weights for w in t if w != 1.0])>0:
            m_t_ft = self.likelihood_teams()
            self.likelihoods = [[ (1/self.weights[e][i] if self.weights[e][i]!=0.0 else inf) * (m_t_ft[e] - self.performance(e).exclude(self.teams[e][i].prior*self.weights[e][i])) for i in range(len(self.teams[e])) ] for e in range(len(self))]
        else:
            self.likelihoods = self.likelihood_analitico()    
       
    def posteriors(self):
        return [[ self.likelihoods[e][i] * self.teams[e][i].prior for i in range(len(self.teams[e]))] for e in range(len(self))]