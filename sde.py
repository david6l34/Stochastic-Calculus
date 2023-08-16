import numpy as np

def brownian_motion(n=100, T=1):
    """Return sample path of brownian motion in [0, T] with n steps"""
    dt = T/(n-1)
    increments = np.random.normal(0, np.sqrt(dt), size=n)
    increments[0] = 0
    B_t = np.cumsum(increments)
    return B_t


class SDE:
    """Base class that represent a SDE"""
    def __init__(self, volatility: callable, drift: callable, d_volatility=None, **kwargs):
        self.volatility = volatility
        self.drift = drift
        self.d_volatility = d_volatility
        self.__dict__.update(**kwargs)

    def sample(self, scheme, x0=0, n=100, T=1, **kwargs):
        return scheme(self, n=n, x0=x0, T=T, **kwargs)


class GeometricBrownian(SDE):
    r"""Geometric brownian motion satisfying the SDE
        .. math:: dS_t=S_t dB_t + (\mu + 1/2)S_t dt
    """
    def __init__(self, mu: float):
        super().__init__(volatility=lambda x: x, drift=lambda x: (mu + 1/2)*x, d_volatility=lambda x:1)
        self.mu = mu

    def analytic(self, x0, n=100, T=1, b=None):
        if b is None:
            b = brownian_motion(n, T)
        t = np.linspace(0, T, n)
        return x0 * np.exp(self.mu * t + b)


class OrnsteinUhlenbeck(SDE):
    r"""
    Ornstein-Uhlenbeck process satisfying the SDE
    .. math:: dX_t = -X_t dt + dB_t
    """
    def __init__(self) :
        super().__init__(volatility=lambda x: 1, drift=lambda x: -x, d_volatility=lambda x:0)


class Diffusion(SDE):
    r"""
    A diffusion process satisfying the SDE
    .. math:: dX_t = \sqrt{1 + {X_t}^2 dB_t} + \sin X_t dB_t
    """
    def __init__(self):
        super().__init__(volatility=lambda x: np.sqrt(1 + x**2), drift=lambda x: np.sin(x), d_volatility=lambda x: 2*x*(1+x**2)**(-0.5))

class CIR(SDE):
    r"""
    The Cox-Ingersoll-Ross (CIR) model satisfying the SDE
    .. math:: dS_t = \sigma \sqrt{S_t} dW_t + (a-bS_t)dt,  S_0>0 and a,b \in R
    """
    def __init__(self, a=1, b=1, sigma=None, d_sigma=None):
        if sigma is not None and d_sigma is None:
             raise ValueError("Please specify d_sigma as well")
        if sigma is None and d_sigma is None:
            sigma = lambda x: 1
            d_sigma = lambda x: 0
        super().__init__(volatility=lambda x: sigma(x)*np.sqrt(x), drift=lambda x: a-b*x,
                         d_volatility=lambda x: d_sigma(x)*np.sqrt(x) + sigma(x)/(2*np.sqrt(x)))

def euler_maruyama(sde: SDE, n=100, x0=1, T=1, b=None):
    volatility = sde.volatility
    drift = sde.drift
    dt = T/(n-1)
    x = np.empty(n)
    x[0] = x0
    if b is None:
        b = brownian_motion(n, T=T)
    if len(b) != len(x):
        raise IndexError(f"Length of Brownian Motion is {len(b)}, doesn't match 1/step_size + 1 = {n}")

    for t in range(1, n):
        increment = volatility(x[t-1]) * (b[t] - b[t-1]) + drift(x[t-1]) * dt
        x[t] = x[t-1] + increment
    return x

def milstein(sde: SDE, n=100, x0=1, T=1, b=None):
    volatility = sde.volatility
    drift = sde.drift
    d_volatility = sde.d_volatility
    dt = T/(n-1)
    x = np.empty(n)
    x[0] = x0
    if b is None:
        b = brownian_motion(n, T=T)
    if len(b) != len(x):
        raise IndexError(f"Length of Brownian Motion is {len(b)}, doesn't match 1/step_size + 1 = {n}")

    for t in range(1, n):
        db = b[t] - b[t-1]
        increment = volatility(x[t-1]) * db \
                    + drift(x[t-1]) * dt \
                    + d_volatility(x[t-1]) * volatility(x[t-1])/2 * (db**2 - dt)
        x[t] = x[t-1] + increment
    return x

