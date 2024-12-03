from ..core.gaussian import Gaussian
from ..core.player import Player
from ..game.game import Game
from .history import History
from ..config import *
from math import inf
import numpy as np

def predict(history, players1, players2, times):
    """Predict outcomes of hypothetical matches between arrays of players at given times.
    
    Args:
        history: Fitted History object containing rating information
        players1: Array of player1 IDs  
        players2: Array of player2 IDs
        times: Array of time points for prediction
        
    Returns:
        Array of p1_win_probs
    """
    def get_ratings(players, times):
        ratings = []
        for player, time in zip(players, times):
            if player not in history.agents:
                ratings.append(Player(Gaussian(MU, SIGMA)))
            else:
                agent = history.agents[player]
                elapsed = time - agent.last_time if agent.last_time != -inf else 0
                ratings.append(Player(agent.receive(elapsed)))
        return ratings

    # Get ratings for all players
    p1_ratings = get_ratings(players1, times)
    p2_ratings = get_ratings(players2, times)
    
    # Create games and get probabilities
    p1_win_probs = np.zeros(len(players1))
    for i, (p1_rating, p2_rating) in enumerate(zip(p1_ratings, p2_ratings)):
        game = Game([[p1_rating], [p2_rating]])
        p1_win_probs[i] = game.evidence
        
    return p1_win_probs