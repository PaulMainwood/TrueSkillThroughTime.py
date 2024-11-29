import math

sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2 * math.pi)
inf = math.inf

def erfc(x):
    z = abs(x)
    t = 1.0 / (1.0 + z / 2.0)
    a = -0.82215223 + t * 0.17087277
    b = 1.48851587 + t * a
    c = -1.13520398 + t * b
    d = 0.27886807 + t * c
    e = -0.18628806 + t * d
    f = 0.09678418 + t * e
    g = 0.37409196 + t * f
    h = 1.00002368 + t * g
    r = t * math.exp(-z * z - 1.26551223 + t * h)
    return r if not(x < 0) else 2.0 - r

def erfcinv(y):
    if y >= 2: return -inf
    if y < 0: raise ValueError('argument must be nonnegative')
    if y == 0: return inf
    if not (y < 1): y = 2 - y
    t = math.sqrt(-2 * math.log(y / 2.0))
    x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)
    for _ in [0, 1, 2]:
        err = erfc(x) - y
        x += err / (1.12837916709551257 * math.exp(-(x**2)) - x * err)
    return x if (y < 1) else -x

def cdf(x, mu=0, sigma=1):
    z = -(x - mu) / (sigma * sqrt2)
    return 0.5 * erfc(z)

def pdf(x, mu, sigma):
    normalizer = (sqrt2pi * sigma)**-1
    functional = math.exp(-((x - mu)**2) / (2*sigma**2))
    return normalizer * functional

def ppf(p, mu, sigma):
    return mu - sigma * sqrt2 * erfcinv(2 * p)

def mu_sigma(tau_, pi_):
    if pi_ > 0.0:
        sigma = math.sqrt(1/pi_)
        mu = tau_ / pi_
    else:
        sigma = inf 
        mu = 0.0
    return mu, sigma

def tau_pi(mu, sigma):
    if sigma > 0.0:
        pi_ = sigma ** -2
        tau_ = pi_ * mu
    else:
        pi_ = inf
        tau_ = inf
    return tau_, pi_