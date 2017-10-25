import networkx as nx
import random
import numpy as np


def _random_subset(seq, m):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    targets = set()
    while len(targets) < m:
        x = random.choice(seq)
        targets.add(x)
    return targets


def dms_graph_basic(N, k0, m=5, seed=None):
    """
    Build DorogovtsevMendesSamukhin graph.

    Graph is 'basic' as we use the case m=m0 (as required for problem sheet m=m0=5).

    Usage: G=dms_graph_basic(N,k0)

    Input:
    N - Number of nodes
    k0 - damping factor

    Output:
    G - A Networkx graph
    """

    if seed is not None:
        random.seed(seed)

    G = nx.complete_graph(m)
    targets = list(range(m))
    repeated_nodes = list(range(m)) * (m - 1 + k0)
    source = m

    while source < N:
        G.add_edges_from(zip([source] * m, targets))
        repeated_nodes.extend(targets)
        repeated_nodes.extend([source] * (m + k0))
        targets = _random_subset(repeated_nodes, m)
        source += 1
    return G


def deg_dist(G):
    """
    Builds the degree dist for a graph G

    Usage: d=deg_dist(G)

    Input:
    G - A graph with N nodes

    Output:
    d - A Numpy array of length N with d[k] = probability of degree k
    """
    N = len(nx.nodes(G))
    degs = nx.degree_histogram(G)  # get degree histogram (list)
    if len(degs) < N:  # standadize length (for when we run many realizations)
        degs = degs + [0] * (N - len(degs))
    degs = np.asarray(degs, dtype=float)  # make numpy array
    degs = degs / np.sum(degs)
    return degs


def knn(G, kmax=200):
    """
    Get the neighbour distribution knn

    Usage: knn=knn(G,kmax)

    Input:
    G - a graph
    kmax - the max degree to consider

    Output:
    knn - a vector of length kmax. NOTE: We should careful to interpret 0's in this vector properly. If knn[k]=0 this just means it is undefined and should be ignored when averaging over reps!!

    """
    N = len(nx.nodes(G))
    knn = np.zeros(kmax)  # the final vector
    A = nx.to_numpy_matrix(G)  # get adj matix as numpy array
    d = np.asarray(list(dict(nx.degree(G)).values()), dtype=float)  # get degrees, convert to numpy, reshape.
    d = np.reshape(d, (1, N))  # So d[i] is degree of node i
    for k in range(kmax):  # loop on k
        delta = np.equal(k, d).astype(int)
        delta = np.reshape(delta, (1, N))  # delta vector. delta[i]=1 if d[i]=k.
        t1 = np.multiply(np.multiply(np.multiply(A, delta.T), d), 1 / d.T)  # numerator (uses broadcasting. Google me!)
        t1 = np.sum(np.sum(t1))
        t2 = np.sum(delta)  # denominator
        if t2 != 0:  # fill in Knn[k,r]
            knn[k] = t1 / t2
    return knn


def flatten(A):
    """
    Flattens an array of dims nxm to a vector of shape 1x(n*m)

    Usage: f=flatten(A)
    """
    return np.reshape(A, (1, -1))


def connected_comps_sizes(G):
    """
    Reverse List of sizes of connectd components
    """
    return [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
