##Libraries
import numpy as np
from copy import deepcopy
from scipy.special import gammaln

try:
    from .utils import create_counts
    from .trees import random_tree
except:
    from utils import create_counts
    from trees import random_tree

"""
ModelTime for bayesian inference using MCMC
"""

class ModelTimeAdj:
    """
    ATTRIBUTS
    n : nombre de noeuds
    r : nombre d'états possibles
    time : nombre d'échantillons temporels
    time_weight : np.array((time,n,n)) of weights
    time_counts : dictionnaire de dictionnaire hyperparamètres N 
    (t : ((u,v) : N_uv ou u : N_u))
    time_struct : dictionnaire des structures des arbres
    (t : structure tuple)
    time_adj : np.array((time,n,n)) the adjacency matrix
    
    PARAMETERS
    ----------
    lambda_1 : weigth of similarity
    """

    def __init__(self, n, time, lambda_1, r=2, s=1):
        self.n = n
        self.time = time
        self.r = r
        
        #TIME_WEIGHT
        time_weight = np.ones((time,n,n)) - np.eye(n)
        self.time_weight = 1 * time_weight
        
        #TIME COUNTS
        counts = {}
        for u in range(1,n+1):
            for v in range(u+1,n+1):
                # On copie le prior de [Schwaller2019]
                counts[(u,v)] = s * 0.5*np.ones((r,r))
        for u in range(1,n+1):
            counts[u] = np.zeros(r)
            # On exploite numpy :)
            if u < n: counts[u] = np.sum(counts[(u,u+1)], axis=1)
            else: counts[u] = np.sum(counts[(1,u)], axis=0)
        time_counts = {t: deepcopy(counts) for t in range(time)}
        self.time_counts = time_counts
        
        #PARAMETERS
        self.lambda_1 = lambda_1
    
    def sample(self, kmax, x):
        """
        Parameters
        ----------
        kmax : len of the result of MCMC with Gibbs method
        x : dicct (t,data(t)) : data used for inference. If None, 
        uses default weights & counts
        
        Returns
        -------
        [Z^(0), Z^(1), Z^(2), Z^(3) ..., Z^(kmax-1)] : 
        List of models of all the states of MCMC
        """
        n = self.n
        time = self.time
        # Calculating the BASAL W
        time_W = np.zeros((time,n,n))
        for t in range(self.time):
            # ATTENTION : il y a un piège ici car les dictionnaires
            # et les numpy arrays sont des objets mutables
            N = create_counts(x[t])
            Nold = self.time_counts[t]
            Nnew = {k: Nold[k] + N[k] for k in N}
            # Mise à jour des comptages
            self.time_counts[t] = Nnew
            # Mise à jour des poids
            lw = {}
            # 1. Calcul des log(Wu)
            for u in range(1,n+1):
                lw[u] = np.sum(gammaln(Nnew[u]) - gammaln(Nold[u]))
            # 2. Calcul des log(Wuv)
            for u in range(1,n+1):
                for v in range(u+1,n+1):
                    lw[u,v] = np.sum(gammaln(Nnew[u,v]) - gammaln(Nold[u,v]))
                    lw[u,v] -= lw[u] + lw[v]
            # 3. Normalisation et calcul des poids
            mlw = gammaln(np.sum(Nnew[1])) - gammaln(np.sum(Nold[1]))
            for u in range(n):
                for v in range(u+1,n):
                    time_W[t][u,v] = np.exp(lw[u+1,v+1] + mlw)
                    time_W[t][v,u] = time_W[t][u,v]
        time_basal = np.zeros((time,n,n))
        for t in range(time):
            time_basal[t] = self.time_weight[t] * time_W[t]
        # Initialize GIBBS
        res = np.zeros((kmax+1,time,n,n), dtype=int)
        ### INITIALISER ICI (OU PAS)
        for k in range(1,kmax+1):
            print(f'Gibbs iteration {k}')
            # Next state of GIBBS
            res[k] = self.gibbs_update(time_basal, res[k-1])
        return res[1:]
    
    def gibbs_update(self, time_basal, time_adj):
        """
        Parameters
        ----------
        self : previous state in the Markov Chain
        x : dicct (t,data(t)) : data used for inference. If None, 
        uses default weights&counts
        
        Returns
        -------
        next_model : Another ModelTime of the next step in the Markov Chain
        """
        # Initialization 
        time, n = self.time, self.n
        time_inter = np.exp(self.H(time_adj))
        times_shuffled = np.arange(time)
        np.random.shuffle(times_shuffled)
        # On initialise la matrice d'adjacence
        Z = np.zeros((time,n,n))
        for i in times_shuffled:
            # print(f't = {i}')
            w = time_basal[i] * time_inter[i]
            # print(w)
            tree = random_tree(n, w)
            for u, children in enumerate(tree):
                if u > 0:
                    for v in children:
                        Z[i,u-1,v-1] = 1
                        Z[i,v-1,u-1] = 1
        return Z

    def H(self, time_adj):
        """
        Returns
        -------
        H: part of the weights due to lambda1 given current Z
        H(t) = lambda1 * Z(1) if t == 0
               lambda1 * Z(time-2) if t == time-1
               lambda1 * Z(t-1) + Z(t+1) else
        """
        lambda_1 = self.lambda_1
        time, n = self.time, self.n
        les_H = np.zeros((time,n,n))
        # On remplit
        les_H[0] = lambda_1 * time_adj[1]
        les_H[time-1] = lambda_1 * time_adj[time-2]
        for t in range(1,time-1):
            les_H[t] = lambda_1 * (time_adj[t-1] + time_adj[t+1])
        return les_H        

    
##TEST
if __name__ == '__main__':
    print("nothing here")
