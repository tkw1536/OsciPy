"""
Code that contains all graph generators
"""

import networkx as nx

def generate_graph(N):
    """
    Generates a graph we do not understand yet. 
    
    Arguments:
        N
            Size of the graph
    Returns:
        a graph
    """
    
    # Start with an empty graph
    graph = nx.Graph()
    
    # number of clusters to create
    clus_num = 4
    
    # for each of the other clusters
    for j in range(clus_num):
        
        # create a new complete graph
        g = nx.complete_graph(N)
        
        # we rename the graph
        nl_map = {}
        for i in range(N):
            nl_map[i] = i + N * j
        g = nx.relabel_nodes(g, nl_map)
        
        # and then we compose
        graph = nx.compose(graph, g)
    
    # add some connections
    # TODO: Generalise this somehow
    graph.add_edges_from([
        (0,4), (1,5),
        (8,12), (9,13),
        (0,12)
    ])

    # and return the graph
    return graph

def generate_ring_graph(N):
    """
    Generates a graph in a ring graph. 
    
    Arguments:
        size
            Size of the graph
    Returns:
        a ring graph with N nodes
    """
    
    # start with an empty graph
    graph = nx.Graph()
    
    # each edge connects to the next one
    for i in range(N):
        graph.add_edge(i, (i+1) % N)
    
    # and return it
    return graph

def generate_paper_graph(_):
    """
    Generates the graph from figure 3 of 
    A. Arenas et al. / Physica D 224 (2006)
    
    Arguments:
        _
            unused
    Returns:
        the given graph
    """
    
    # create a new graph
    graph = nx.Graph()
    
    # hard-coded edges
    graph.add_edges_from([
        (1,5),(5,6),(6,1),
        (0,3),(3,4),(4,0),
        (2,7),(7,8),(8,2),
        (1,0),(5,0),(6,0),
        (2,0),(7,0),(8,0)
    ])
    
    # and return the graph
    return graph
