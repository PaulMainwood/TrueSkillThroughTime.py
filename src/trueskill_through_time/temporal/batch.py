class Batch(object):
    def __init__(self, composition, results = [] , time = 0, agents = dict(), p_draw=0.0, weights = []):
        if (len(results)>0) and (len(composition)!= len(results)): raise ValueError("(len(results)>0) and (len(composition)!= len(results))")
        if (len(weights)>0) and (len(composition)!= len(weights)):raise ValueError("(len(weights)>0) & (len(composition)!= len(weights))")

        this_agents = set( [a for teams in composition for team in teams for a in team ] )
        elapsed = dict([ (a,  compute_elapsed(agents[a].last_time, time) ) for a in this_agents ])
        
        self.skills = dict([ (a, Skill(agents[a].receive(elapsed[a]) ,Ninf ,Ninf , elapsed[a])) for a in this_agents  ])
        self.events = [Event([Team([Item(composition[e][t][a], Ninf) for a in range(len(composition[e][t])) ], results[e][t] if len(results) > 0 else len(composition[e]) - t - 1  ) for t in range(len(composition[e])) ],0.0, weights if not weights else weights[e]) for e in range(len(composition) )]
        self.time = time
        self.agents = agents
        self.p_draw = p_draw
        self.iteration()
    
    def __repr__(self):
        return "Batch(time={}, events={})".format(self.time,self.events)
    def __len__(self):
        return len(self.events)
    def add_events(self, composition, results = []):
        b=self
        this_agents = set( [a for teams in composition for team in teams for a in team ] )
        for a in this_agents:
            elapsed = compute_elapsed(b.agents[a].last_time , b.time )  
            if a in b.skills:
                b.skills[a] = Skill(b.agents[a].receive(elapsed) ,Ninf ,Ninf , elapsed)
            else:
                b.skills[a].elapsed = elapsed
                b.skills[a].forward = b.agents[a].receive(elapsed)
        _from = len(b)+1
        for e in range(len(composition)):
            event = Event([Team([Item(composition[e][t][a], Ninf) for a in range(len(composition[e][t]))], results[e][t] if len(results) > 0 else len(composition[e]) - t - 1 ) for t in range(len(composition[e])) ] , 0.0, weights if not weights else weights[e])
            b.events.append(event)
        b.iteration(_from)
    def posterior(self, agent):
        return self.skills[agent].likelihood*self.skills[agent].backward*self.skills[agent].forward
    def posteriors(self):
        res = dict()
        for a in self.skills:
            res[a] = self.posterior(a)
        return res
    def within_prior(self, item):
        r = self.agents[item.name].player
        mu, sigma = self.posterior(item.name)/item.likelihood
        res = Player(Gaussian(mu, sigma), r.beta, r.gamma)
        return res
    def within_priors(self, event):#event=0
        return [ [self.within_prior(item) for item in team.items ] for team in self.events[event].teams ]
    def iteration(self, _from=0):#self=b
        for e in range(_from,len(self)):#e=0
            teams = self.within_priors(e)
            result = self.events[e].result
            weights = self.events[e].weights
            g = Game(teams, result, self.p_draw, weights)
            for (t, team) in enumerate(self.events[e].teams):
                for (i, item) in enumerate(team.items):
                    self.skills[item.name].likelihood = (self.skills[item.name].likelihood / item.likelihood) * g.likelihoods[t][i]
                    item.likelihood = g.likelihoods[t][i]
            self.events[e].evidence = g.evidence
    def convergence(self, epsilon=1e-6, iterations = 20):
        step, i = (inf, inf), 0
        while gr_tuple(step, epsilon) and (i < iterations):
            old = self.posteriors().copy()
            self.iteration()
            step = dict_diff(old, self.posteriors())
            i += 1
        return i
    def forward_prior_out(self, agent):
        return self.skills[agent].forward * self.skills[agent].likelihood
    def backward_prior_out(self, agent):
        N = self.skills[agent].likelihood*self.skills[agent].backward
        return N.forget(self.agents[agent].player.gamma, self.skills[agent].elapsed) 
    def new_backward_info(self):
        for a in self.skills:
            self.skills[a].backward = self.agents[a].message
        return self.iteration()
    def new_forward_info(self):
        for a in self.skills:
            self.skills[a].forward = self.agents[a].receive(self.skills[a].elapsed) 
        return self.iteration()