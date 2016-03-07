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

def plot_evolutions(sols, ts, join_events, ax):
    """ Plot system evolution
    """
    for i, sol in enumerate(sols):
        ax.plot(sol, label=r'$f_{{{}}}$'.format(i))

    for jev_x in join_events:
        ax.axvline(jev_x, color='r', alpha=0.5, ls='dashed')

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
    node_num = sols.shape[1]
    mat = np.ones((node_num, node_num))
    join_events = []
    already_clustered = []
    cluster_groups = collections.defaultdict(list)

    res = []
    step_size = 50
    for series in sols.T:#np.diff(sols, axis=0).T:
        res.append([])
        for i in range(0, len(series), step_size):
            res[-1].append(np.mean(series[i:i+step_size]))
    res = np.array(res).T
    max_len = res.shape[0]

    for t, state in enumerate(res):
        for i, i_val in enumerate(state):
            if i in already_clustered: continue
            for j, j_val in enumerate(state):
                if i >= j or j in already_clustered: continue

                if np.isclose(i_val, j_val):
                    print('->', i, j, t, max_len)
                    for n in [j]:
                        mat[i, n] = t / max_len
                        mat[n, i] = mat[i, n]

                    already_clustered.append(j)
                    cluster_groups[i].append(j)
                    join_events.append(t * step_size)

    return mat, join_events

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

    """ Show system evolution for different adjacency matrices
    """
    graph = generate_graph_paper() #generate_graph(size)

    adjacency_matrix = nx.to_numpy_matrix(graph)
    omega_vec = np.ones((len(graph.nodes()),)) * 0.3

    sols, ts = solve_system(omega_vec, adjacency_matrix)
    mat, jevs = generate_sync_matrix(sols)

    plot_result(graph, mat, (sols.T, ts, jevs))

def main():
    """ General interface
    """
    simulate_system(4)

if __name__ == '__main__':
    main()
