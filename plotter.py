"""
All functions related to plotting
"""

import numpy as np
import networkx as nx

import matplotlib.pylab as plt
import matplotlib as mpl


def plot_matrix(mat, ax):
    """ Plot system evolution
    """
    im = ax.imshow(
        mat,
        interpolation='nearest', cmap=plt.cm.coolwarm)
    plt.colorbar(im, ax=ax)

    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$j$')

    ax.set_title('Sign-switch of dynamic connectivity matrix')

def plot_graph(graph, ax):
    """ Plot graph
    """
    # generate some node properties
    labels = {}
    for n in graph.nodes():
        labels[n] = n

    # compute layout
    pos = nx.nx_pydot.graphviz_layout(graph, prog='neato')

    # draw graph
    nx.draw(
        graph, pos,
        node_color='lightskyblue', node_size=800,
        font_size=20,
        ax=ax)
    nx.draw_networkx_labels(graph, pos, labels)

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

    ax.set_xlabel(r'$t$')
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

    plt.tight_layout()
    fig.savefig('result.pdf', dpi=300)
    fig.savefig('foo.png')
