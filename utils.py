"""
Commonly used functions
"""

import numpy as np
import numpy.random as npr

from scipy.integrate import odeint


def generate_system(omega_vec, A):
    """ Generate generalized Kuramoto model
    """
    def func(state, t=0):
        ode = []
        for i, o in enumerate(omega_vec):
            ode.append(
                o + np.sum(
                    [A[i,j] * np.sin(state[j] - state[i])
                        for j in range(len(omega_vec))])
            )
        return np.array(ode)
    return func

def solve_system(O, A, tmax=30, dt=0.01):
    """ Solve particular configuration
    """
    func = generate_system(O, A)

    ts = np.arange(0, tmax, dt)
    init = npr.uniform(0, 2*np.pi, size=O.shape)

    sol = odeint(func, init, ts)
    sol %= 2*np.pi

    return sol, ts
