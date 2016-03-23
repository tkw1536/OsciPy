"""
Investigate how long synchronization between individual cells takes
"""

import numpy as np
import networkx as nx

from tqdm import trange

from utils import solve_system, DictWrapper
from plotter import *
from generators import *
from investigations import *


def simulate_system(size, reps=50):
    """ Have fun :-)
    """
    # setup network
    graph = generate_graph(size)
    investigate_laplacian(graph)

    # setup dynamical system
    omega = 0.2
    OMEGA = 3
    dim = len(graph.nodes())
    system_config = DictWrapper({
        'A': nx.to_numpy_matrix(graph),
        'B': np.ones((dim,)),
        'o_vec': np.ones((dim,)) * omega,
        'Phi': lambda t: OMEGA * t
    })

    # solve system on network
    corr_mats = []
    var_sers = []
    for _ in trange(reps):
        sols, ts = solve_system(system_config)

        cmat = compute_correlation_matrix(sols.T)
        vser = compute_cluster_num(sols.T, len(graph.nodes()))

        corr_mats.append(cmat)
        var_sers.append(vser)
    corr_mats = np.array(corr_mats)
    var_sers = np.array(var_sers)

    # compute synchronization offsets
    mean_time = np.mean(corr_mats, axis=0)
    dcm = compute_dcm(mean_time)
    sync_time = compute_sync_time(dcm, ts)
    #print(sync_time)

    # further investigations
    mean_var = np.mean(var_sers, axis=0)

    # plot results
    data = DictWrapper({
        'graph': graph,
        'syncs': sync_time,
        'cmats': mean_time,
        'sol': sols.T,
        'ts': ts,
        'vser': mean_var
    })
    plot_result(data)

def main():
    """ General interface
    """
    simulate_system(4)

if __name__ == '__main__':
    main()
