"""
Investigate how long synchronization between individual cells takes
"""

import sys

import numpy as np
import networkx as nx

from tqdm import trange

from utils import solve_system, DictWrapper

import plotter
import generators
import investigations

import matplotlib.pylab as plt

def setup_system(size=4):
    """ All system configurations go here
    """
    
    # setup network
    graph = generators.generate_graph(size)
    investigations.investigate_laplacian(graph)

    # setup dynamical system
    omega = 0.2
    OMEGA = 3
    dim = len(graph.nodes())
    system_config = DictWrapper({
        'A': nx.to_numpy_matrix(graph),
        'B': np.ones((dim,)),
        'o_vec': np.ones((dim,)) * omega,
        'Phi': lambda t: OMEGA * t,
        'OMEGA': OMEGA
    })

    return DictWrapper({
        'graph': graph,
        'system_config': system_config
    })

def simulate_system(bundle, reps=50):
    """ Generate data from system setup
    """
    # lonely investigation :'(
    investigations.investigate_laplacian(bundle.graph)

    # solve system on network
    corr_mats = []
    var_sers = []
    all_sols = []
    for _ in trange(reps):
        sols, ts = solve_system(bundle.system_config)

        cmat = investigations.compute_correlation_matrix(sols)
        vser = investigations.compute_cluster_num(sols, len(bundle.graph.nodes()))

        corr_mats.append(cmat)
        var_sers.append(vser)
        all_sols.append(sols)

    bundle['all_sols'] = all_sols
    bundle['corr_mats'] = np.array(corr_mats)
    bundle['var_sers'] = np.array(var_sers)
    bundle['ts'] = ts

    return bundle

def handle_solution(bundle):
    """ Investigate previously generated data
    """
    # compute synchronization offsets
    mean_time = np.mean(bundle.corr_mats, axis=0)
    dcm = investigations.compute_dcm(mean_time)
    sync_time = investigations.compute_sync_time(dcm, bundle.ts)
    #print(sync_time)

    # further investigations
    mean_var = np.mean(bundle.var_sers, axis=0)
    investigations.reconstruct_coupling_params(bundle)

    # plot results
    data = DictWrapper({
        'graph': bundle.graph,
        'syncs': sync_time,
        'cmats': bundle.corr_mats,
        'sols': bundle.all_sols,
        'ts': bundle.ts,
        'vser': mean_var
    })
    plotter.plot_result(data)

def main(args):
    """
    Main entry point. 
    
    Arguments:
        args
            Arguments parsed to this function
    Returns:
        an integer representing the return code
    """

    bundle = setup_system()

    bundle = simulate_system(bundle)
    handle_solution(bundle)
    
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))