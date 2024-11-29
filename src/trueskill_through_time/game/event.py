class Event:
    def __init__(self, teams, evidence, weights):
        self.teams = teams
        self.evidence = evidence
        self.weights = weights
        
    def __repr__(self):
        return f"Event({self.names}, {self.result})"
    
    @property
    def names(self):
        return [[item.name for item in team.items] for team in self.teams]
    
    @property
    def result(self):
        return [team.output for team in self.teams]
        
def get_composition(events):
    return [ [[ it.name for it in t.items] for t in e.teams] for e in events]

def get_results(events):
    return [ [t.output for t in e.teams ] for e in events]

def compute_elapsed(last_time, actual_time):
    return 0 if last_time == -inf  else ( 1 if last_time == inf else (actual_time - last_time))