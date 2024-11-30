from ..core.gaussian import Gaussian
from ..core.player import Player
from ..game.game import Game
from .history import History
from ..config import *
from math import inf

def predict(history, player1, player2, time):
    """Predict outcome of a hypothetical match between two players at given time.
    
    Args:
        history: Fitted History object containing rating information
        player1: String identifier for first player
        player2: String identifier for second player  
        time: Time point for prediction
        
    Returns:
        Tuple of (p1_win_prob, p2_win_prob)
    """
    # Get the most recent rating before 'time' for each player
    def get_rating(player):
        if player not in history.agents:
            # New player - use default rating
            return Player(Gaussian(MU, SIGMA))
        
        # Get player's rating at specified time
        agent = history.agents[player]
        elapsed = time - agent.last_time if agent.last_time != -inf else 0
        return Player(agent.receive(elapsed))

    p1_rating = get_rating(player1)
    p2_rating = get_rating(player2)
    
    # Create single-game History to compute win probability
    game = Game([[p1_rating], [p2_rating]])
    
    # Extract win probability from game likelihood
    p1_win_prob = game.evidence
    
    return (p1_win_prob, 1 - p1_win_prob)