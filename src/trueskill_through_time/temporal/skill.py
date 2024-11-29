class Skill:
    def __init__(self, forward=None, backward=None, likelihood=None, elapsed=0):
        self.forward = forward or Gaussian(0, inf)
        self.backward = backward or Gaussian(0, inf)
        self.likelihood = likelihood or Gaussian(0, inf)
        self.elapsed = elapsed

__all__ = ['Skill']