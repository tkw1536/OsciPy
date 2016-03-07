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

def simulate_system(size, reps=10):
    """ Show system evolution for different adjacency matrices
    """
    graph = generate_graph_paper() #generate_graph(size)

    adjacency_matrix = nx.to_numpy_matrix(graph)
    omega_vec = np.ones((len(graph.nodes()),)) * 2

    mats = []
    for _ in trange(reps):
        sols, ts = solve_system(omega_vec, adjacency_matrix)
        corr_mat = compute_correlation_matrix(sols.T)
        mats.append(corr_mat)
    mats = np.array(mats)

    mean_time = np.mean(mats, axis=0)
    time_sum = np.sum(mean_time, axis=2)

    plot_result(graph, time_sum, mean_time, sols.T, ts)

def main():
    """ General interface
    """
    simulate_system(4)

if __name__ == '__main__':
    main()
