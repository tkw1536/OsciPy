"""
All graph generators
"""

import networkx as nx


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
