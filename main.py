"""
Investigate how long synchronization between individual cells takes
"""

import numpy as np
import networkx as nx

from tqdm import trange

from utils import solve_system
from plotter import *
from generators import *


def compute_correlation_matrix(sols):
    """ Compute correlations as described in paper
    """
    cmat = np.empty((sols.shape[0], sols.shape[0], sols.shape[1]))
    for i, sol in enumerate(sols):
        for j, osol in enumerate(sols):
            cmat[i, j] = np.cos(sol - osol)
    return cmat

def compute_dcm(corr_mat):
    """ Compute dynamic connectivity matrix
    """
    dcm = []
    for thres in [0.9]: #np.linspace(0, 1, 5):
        cur_mat = corr_mat > thres
        dcm.append(cur_mat)

    return dcm[0]

def compute_sync_time(dcm):
    """ Compute time it takes to synchronize from DCM
    """
    print(dcm.shape)
    print(dcm.T.shape)

    sync_time = -np.ones((dcm.shape[0], dcm.shape[0]))
    for t, state in enumerate(dcm.T):
        #inds = set([(i, j) for i,j in zip(*np.nonzero(state))])
        #ncl = set([(i, j) for i,j in np.argwhere(sync_time < 0)])
        #sel = list(zip(*inds.union(ncl)))
        #sync_time[sel] = t

        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i,j] == 1 and sync_time[i,j] < 0:
                    sync_time[i,j] = t

    return sync_time

def simulate_system(size, reps=15):
    """ Have fun :-)
    """
    graph = generate_graph(size)

    adjacency_matrix = nx.to_numpy_matrix(graph)
    omega_vec = np.ones((len(graph.nodes()),)) * 2

    mats = []
    for _ in trange(reps):
        sols, ts = solve_system(omega_vec, adjacency_matrix)
        corr_mat = compute_correlation_matrix(sols.T)
        mats.append(corr_mat)
    mats = np.array(mats)

    mean_time = np.mean(mats, axis=0) # average all repretitions
    dcm = compute_dcm(mean_time)
    sync_time = compute_sync_time(dcm)
    print(sync_time)

    plot_result(graph, sync_time, mean_time, sols.T, ts)

def main():
    """ General interface
    """
    simulate_system(4)

if __name__ == '__main__':
    main()
