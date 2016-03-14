"""
Investigate how long synchronization between individual cells takes
"""

import numpy as np
import networkx as nx

from tqdm import trange

from utils import solve_system, DictWrapper
from plotter import *
from generators import *


def compute_correlation_matrix(sols):
    """ Compute pairwise node-correlations of solution
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

def compute_sync_time(dcm, ts):
    """ Compute time it takes to synchronize from DCM
    """
    sync_time = -np.ones((dcm.shape[0], dcm.shape[0]))
    for t, state in enumerate(dcm.T):
        inds = np.argwhere((state == 1) & (sync_time < 0))
        sync_time[tuple(inds.T)] = ts[t]
    return sync_time

def compute_cluster_num(sols, graph_size, thres=.05):
    """ Compute pairwise node-variances of solution
    """
    series = []
    for t, state in enumerate(sols.T):
        # compute sine sum
        sin_sum = 0
        for i_theta in state:
            for j_theta in state:
                sin = np.sin(i_theta - j_theta)**2
                if sin > thres:
                    sin_sum += sin

        # compute actual value 'c'
        c = graph_size**2 / (graph_size**2 - 2 * sin_sum)

        series.append(c)
    return series

def investigate_laplacian(graph):
    """ Compute Laplacian
    """
    w = nx.laplacian_spectrum(graph)

    pairs = []
    for i, w in enumerate(sorted(w)):
        if abs(w) < 1e-5: continue
        inv_w = 1 / w
        pairs.append((inv_w, i))

    plt.figure()
    plt.scatter(*zip(*pairs))
    plt.xlabel(r'$\frac{1}{\lambda_i}$')
    plt.ylabel(r'rank index')
    plt.savefig('le_spectrum.pdf')

def simulate_system(size, reps=50):
    """ Have fun :-)
    """
    graph = generate_graph(size)
    investigate_laplacian(graph)

    adjacency_matrix = nx.to_numpy_matrix(graph)
    omega_vec = np.ones((len(graph.nodes()),)) * 2

    corr_mats = []
    var_sers = []
    for _ in trange(reps):
        sols, ts = solve_system(omega_vec, adjacency_matrix)

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
