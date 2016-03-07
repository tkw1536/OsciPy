"""
Investigate how long synchronization between individual cells takes
"""

import io

import numpy as np
import networkx as nx

import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.image as mpimg

from tqdm import trange

from utils import solve_system


def plot_matrix(mat, ax):
    """ Plot system evolution
    """
    im = ax.imshow(
        mat,
        interpolation='nearest', cmap=plt.cm.coolwarm)
    plt.colorbar(im, ax=ax)

    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$j$')

def plot_graph(graph, ax):
    """ Plot graph
    """
    pydot_graph = nx.nx_pydot.to_pydot(graph)
    png_str = pydot_graph.create_png(prog=['dot', '-Gdpi=300'])
    img = mpimg.imread(io.BytesIO(png_str))

    ax.imshow(img, aspect='equal')
    ax.axis('off')

def plot_evolutions(sols, ts, ax):
    """ Plot system evolution
    """
    for i, sol in enumerate(sols):
        ax.plot(ts, sol, label=r'$f_{{{}}}$'.format(i))

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\Theta_i$')

    ax.set_ylim((0, 2*np.pi))
    #ax.legend(loc='best')

def plot_correlation_matrix(cmat, ts, ax):
    """ Plot individual correlation matrix
    """
    for i, row in enumerate(cmat):
        for j, sol in enumerate(row):
            ax.plot(ts, sol, label='{},{}'.format(i, j))

    ax.set_xlabel('t')
    ax.set_ylabel(r'$\langle \cos \left(\Theta_i(t) - \Theta_j(t)\right)\rangle$')
    ax.set_ylim((-1, 1.1))

def plot_result(graph, mat, cmat, sols, ts):
    """ Plot final result
    """
    fig = plt.figure(figsize=(30, 10))
    gs = mpl.gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2])

    plot_graph(graph, plt.subplot(gs[:, 0]))
    plot_matrix(mat, plt.subplot(gs[:, 1]))
    plot_evolutions(sols, ts, plt.subplot(gs[0, 2]))
    plot_correlation_matrix(cmat, ts, plt.subplot(gs[1, 2]))

    fig.savefig('result.pdf', dpi=300)
    fig.savefig('foo.png')

def compute_correlation_matrix(sols):
    """ Compute correlations as described in paper
    """
    cmat = np.empty((sols.shape[0], sols.shape[0], sols.shape[1]))
    for i, sol in enumerate(sols):
        for j, osol in enumerate(sols):
            cmat[i, j] = np.cos(sol - osol)
    return cmat

def generate_graph(size):
    """ Generate graph with clusters
    """
    clus_num = 4
    graph = nx.complete_graph(size)

    for j in range(1, clus_num):
        g = nx.complete_graph(size)
        nl_map = {}
        for i in range(size):
            nl_map[i] = i + size * j
        g = nx.relabel_nodes(g, nl_map)

        graph = nx.compose(graph, g)

    graph.add_edges_from([
        (0,4), (1,5),
        (8,12), (9,13),
        (0,12)
    ])

    return graph

def generate_graph_paper():
    graph = nx.Graph()
    graph.add_edges_from([
        (2,6),(6,7),(7,2),
        (1,4),(4,5),(5,1),
        (3,8),(8,9),(9,3),
        (2,1),(6,1),(7,1),
        (3,1),(8,1),(9,1)
    ])
    return graph

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
