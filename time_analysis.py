"""
Investigate how long synchronization between individual cells takes
"""

import io

import numpy as np
import networkx as nx

import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.image as mpimg

from utils import solve_system


def plot_matrix(mat, ax):
    """ Plot system evolution
    """
    im = ax.imshow(mat, interpolation='nearest')
    plt.colorbar(im, ax=ax)

    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$j$')

    ax.set_title('Synchronization duration')

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
    ax.legend(loc='best')

def plot_result(graph, mat, evo):
    """ Plot final result
    """
    fig = plt.figure(figsize=(30, 10))
    gs = mpl.gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2])

    plot_graph(graph, plt.subplot(gs[0]))
    plot_matrix(mat, plt.subplot(gs[1]))
    plot_evolutions(*evo, plt.subplot(gs[2]))

    fig.savefig('result.pdf', dpi=300)
    fig.savefig('foo.png')

def generate_sync_matrix(sols):
    """ Generate matrix with synchronization time in each cell
    """
    max_len = len(sols[0])

    mat = []
    for sol in sols:
        mat.append([])
        for osol in sols:
            diff = abs(sol - osol)
            ind = None

            step_size = 50
            for i in range(0, len(diff), step_size):
                chunk = diff[i:i+step_size]
                if np.isclose(np.sum(chunk), 0, atol=0.05):
                    ind = i
                    break

            if ind is None:
                ind = max_len

            dur = ind / max_len
            mat[-1].append(dur)

    return np.array(mat)

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

def simulate_system(size):
    """ Show system evolution for different adjacency matrices
    """
    graph = generate_graph(size)

    adjacency_matrix = nx.to_numpy_matrix(graph)
    omega_vec = np.ones((len(graph.nodes()),)) * 0.3

    sols, ts = solve_system(omega_vec, adjacency_matrix)
    mat = generate_sync_matrix(sols.T)

    plot_result(graph, mat, (sols.T, ts))

def main():
    """ General interface
    """
    simulate_system(4)

if __name__ == '__main__':
    main()
