"""
Bundle all functions together which investigate interesting properties
"""

import random
import itertools

import numpy as np
import networkx as nx
import matplotlib.pylab as plt


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

def reconstruct_coupling_params(bundle):
    """ Try to reconstruct A and B from observed data
    """
    c = bundle.system_config
    sol = bundle.single_sol.T
    dim = c.A.shape[0]**2 + c.A.shape[0]

    # select time points to sample at
    sample_size = 200
    t_points = random.sample(
        range(int(3/4*len(bundle.ts)), len(bundle.ts)), sample_size)

    # create coefficient matrix
    a = np.empty((sample_size, dim))#, dtype='|S5')
    subs = list(itertools.product(range(c.A.shape[0]), repeat=2))
    for i in range(a.shape[0]):
        t = t_points.pop()
        theta = sol[t]

        for j in range(a.shape[1]):
            if j < c.A.shape[0]**2: # fill A_ij
                si, sj = subs[j]
                a[i, j] = np.cos(theta[si] - theta[sj])
                #a[i, j] = 'A_{}{}'.format(si, sj)
            else: # fill B_i
                si = j - c.A.shape[0]**2

                if i % c.A.shape[0] == j % c.A.shape[0]:
                    a[i, j] = np.sin(c.Phi(t) - theta[si])
                    #a[i, j] = 'B_{}'.format(si)
                else:
                    a[i, j] = 0

    # create LHS vector
    b = np.ones(sample_size) * (c.OMEGA - c.o_vec[0])

    # solve system
    x = np.linalg.lstsq(a, b)[0]

    #print(a, b, x)
    print('Original B:', c.B)
    print('Reconstructed B:', x[-2:])
