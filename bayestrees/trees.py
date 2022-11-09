"""
Generate random trees from the weighted uniform distribution
"""
import numpy as np

def random_step(state, w):
    """
    Make one step of the random walk on the weighted graph defined by w.
    NB: here we construct an in-tree so all directions are reversed.
    """
    p = w[:,state]/np.sum(w[:,state])
    return np.dot(np.arange(p.size), np.random.multinomial(1, p))

def loop_erasure(path):
    """
    Compute the loop erasure of a given path.
    """
    if path[0] == path[-1]: return [path[0]]
    else: i = np.max(np.arange(len(path))*(np.array(path)==path[0]))
    if path[i+1] == path[-1]: return [path[0], path[i+1]]
    else: return [path[0]] + loop_erasure(path[i+1:])

def random_spanning_tree(w):
    """
    Generate a random spanning tree rooted in node 0 from the uniform
    distribution with weights given by matrix w (using Wilson's method).
    """
    n = w.shape[0]
    tree = [[] for i in range(n)]
    v = {0} # Vertices of the tree
    r = list(range(1,n)) # Remaining vertices
    while len(r) > 0:
        state = r[0]
        path = [state]
        # compute a random path that reaches the current tree
        while path[-1] not in v:
            state = random_step(path[-1], w)
            path.append(state)
        path = loop_erasure(path)
        # Append the loop-erased path to the current tree
        for i in range(len(path)-1):
            v.add(path[i])
            r.remove(path[i])
            tree[path[i+1]].append(path[i])
    for i in range(n): tree[i].sort()
    return tuple([tuple(tree[i]) for i in range(n)])

# Main functions
def random_tree(n, weight=None):
    """
    Generate a random spanning tree with vertices 1,...,n and root 1.
    NB: NEW VERSION -> SPANNING FORESTS
    """
    w = np.ones((n+1,n+1))
    if weight is not None:
        w[1:,1:] = weight
    w[:,0] = 0
    w = w - np.diag(np.diag(w))
    return random_spanning_tree(w)

def prob_edges(n, weight):
    """
    Compute edge probabilities given edge weights.
    NB: NEW VERSION -> SPANNING FORESTS
    """
    # Compute the laplacian matrix
    l = np.diag(np.sum(weight, axis=1)) - weight
    # Forest matrix q
    q = np.linalg.inv(np.eye(n) + l)
    # Intermediary matrix m
    m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            m[i,j] = q[i,i] + q[j,j] - 2*q[i,j]
    return weight * m

def random_tree_old(n, weight=None):
    """
    Generate a random spanning tree with vertices 1,...,n and root 1.
    NB: OLD VERSION -> SPANNING TREES
    """
    w = np.ones((n+1,n+1))
    if weight is not None:
        w[1:,1:] = weight
    w[:,0] = 0
    w = w - np.diag(np.diag(w))
    w[0,2:] = 0 # Edge 0 -> 1 only
    return random_spanning_tree(w)

def prob_edges_old(n, weight=None):
    """
    Compute edge probabilities given edge weights.
    NB: OLD VERSION -> SPANNING TREES
    """
    if weight is None:
        weight = np.ones((n,n)) - np.eye(n)
    # Compute the laplacian matrix
    l = np.diag(np.sum(weight, axis=1)) - weight
    # Intermediary matrix q
    q = np.zeros((n,n))
    q[1:,1:] = np.linalg.inv(l[1:,1:])
    # Intermediary matrix m
    m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            m[i,j] = q[i,i] + q[j,j] - 2*q[i,j]
    return weight * m

def prob_edges_weight(n, dic_weight):
    """
    Dictionnaire donnant la probabilité de chaque arrête.
    """
    # Matrice des poids
    weight = np.zeros((n,n))
    for (u,v), w in dic_weight.items():
        weight[u-1,v-1], weight[v-1,u-1] = w, w
    p = prob_edges(n, weight)
    return {(u,v): p[u-1,v-1] for u,v in dic_weight}

def get_probas_Gibbs(n, time, kmax, simu_gibbs):
    """
    Parameters
    ----------
    simu_gibbs : result of sample(self, kmax, x) : list of ModelTimeAdj

    Returns
    -------
    res : List of size "time" where res[t] is a dict {(i,j) : array of size 
                                                       kmax showing p_ij(t) by 
                                                       CLT}
    """
    res = []
    for t in range(time):
        dict = {(u,v): np.zeros(kmax) for u in range(1,n+1) for v in range(u+1,n+1)}
        for k in range(kmax):
            probas = prob_edges_weight(n, simu_gibbs[k].time_weight[t])
            for u in range(1,n+1):
                for v in range(u+1,n+1):
                    dict[u,v][k] = probas[u,v]
        res.append(dict)
    return res
            
    
# Tests
if __name__ == '__main__':
    n = 20
    tree = random_tree(n)
    print(tree)

    # # Visualization test
    # import matplotlib.pyplot as plt
    # import networkx as nx
    # G = nx.Graph()
    # G.add_nodes_from(range(1,n+1))
    # for i, targets in enumerate(tree):
    #     for j in targets:
    #         if i > 0: G.add_edge(i,j)
    # # Figure
    # fig = plt.figure(figsize=(8,8))
    # # Layout graphs with positions using graphviz neato
    # pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    # # Color nodes the same in each connected subgraph
    # C = (G.subgraph(c) for c in nx.connected_components(G))
    # for g in C:
    #     c = [0] * nx.number_of_nodes(g)
    #     nx.draw(g, pos, node_size=500, node_color='lightgray',
    #         with_labels=True, linewidths=1, edgecolors='gray',
    #         verticalalignment='center_baseline')
    # fig.savefig('test_tree.pdf')
