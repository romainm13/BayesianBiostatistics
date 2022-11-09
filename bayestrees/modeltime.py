##Libraries
import numpy as np
from copy import deepcopy
from scipy.special import gammaln

try:
    from .utils import create_counts
    from .trees import prob_edges_weight
except:
    from utils import create_counts
    from trees import prob_edges_weight

"""
ModelTime for bayesian inference using MCMC
"""

class ModelTime:
    """
    ATTRIBUTS
    n : nombre de noeuds
    r : nombre d'états possibles
    time : nombre d'échantillons temporels
    time_weight : dictionnaire de dictionnaire hyperparamètres Betas 
    (t : ((u,v) : Beta_uv))
    time_counts : dictionnaire de dictionnaire hyperparamètres N 
    (t : ((u,v) : N_uv ou u : N_u))
    
    PARAMETERS
    ----------
    lambda_1 : weigth of similarity
    """

    def __init__(self, n, time, lambda_1, r=2):
        self.n = n
        self.time = time
        self.r = r
        
        #TIME_WEIGHT
        time_weight = {t : {(u,v): 1 for u in range(1,n+1) for v in 
                            range(u+1,n+1)} for t in range(time)}
        self.time_weight = time_weight
        
        #TIME COUNTS
        counts = {}
        for u in range(1,n+1):
            for v in range(u+1,n+1):
                # On copie le prior de [Schwaller2019]
                counts[(u,v)] = 0.5*np.ones((r,r))
        for u in range(1,n+1):
            counts[u] = np.zeros(r)
            # On exploite numpy :)
            if u < n: counts[u] = np.sum(counts[(u,u+1)], axis=1)
            else: counts[u] = np.sum(counts[(1,u)], axis=0)
        time_counts = {t : deepcopy(counts) for t in range(time)}
        self.time_counts = time_counts
        
        #PARAMETERS
        self.lambda_1 = lambda_1
    
    def sample(self, kmax, x):
        """
        Parameters
        ----------
        kmax : numbers of steps of MCMC with Gibbs method
        x : dicct (t,data(t)) : data used for inference. If None, 
        uses default weights&counts
        
        Returns
        -------
        [model^(1), model^(2), model^(3), model^(4) ..., model^(kmax)] : 
        List of ModelTime objects all the states of MCMC
        """
        model_ini = self
        res = [model_ini]
        for k in range(kmax):
            next_state = res[k].gibbs_update(x) #next state of the MCMC
            res.append(next_state)
        return res
    
    def gibbs_update(self, x):
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
        #Initialization 
        next_model = ModelTime(self.n, self.time, self.lambda_1)
        
        times_shuffled = np.arange(next_model.time)
        np.random.shuffle(times_shuffled)
        for i in times_shuffled:
            #Update N & W at t = i
            n,weight = next_model.n, next_model.time_weight[i]
            # ATTENTION : il y a un piège ici car les dictionnaires
            # et les numpy arrays sont des objets mutables
            N = create_counts(x[i])
            Nold = next_model.time_counts[i]
            Nnew = {k: Nold[k] + N[k] for k in N}
            # Mise à jour des comptages
            next_model.time_counts[i] = Nnew
            # Mise à jour des poids
            # ATTENTION : 2ème piège, les poids sont définis à constante
            # multiplicative près, et il semble que la normalisation soit
            # cruciale avant l'exponentielle pour eviter weight = 0...
            lw = {}
            # 1. Calcul des log(Wu)
            for u in range(1,n+1):
                lw[u] = np.sum(gammaln(Nnew[u]) - gammaln(Nold[u]))
            # 2. Calcul des log(Wuv)
            for u, v in weight:
                lw[u,v] = np.sum(gammaln(Nnew[u,v]) - gammaln(Nold[u,v]))
                lw[u,v] -= lw[u] + lw[v]
            # 3. Normalisation et calcul des poids
            mlw = np.mean([lw[u,v] for u, v in weight])
            for u, v in weight:
                weight[u,v] *= np.exp(lw[u,v] - mlw)
                # NEW FOR ModelTime
                if i == 0:
                    new = prob_edges_weight(self.n,next_model.time_weight[1])[u,v]
                elif i == self.time-1:
                    new = prob_edges_weight(self.n,next_model.time_weight[self.time-1])[u,v]
                else:
                    new1 = prob_edges_weight(self.n,next_model.time_weight[i-1])[u,v]
                    new2 = prob_edges_weight(self.n,next_model.time_weight[i+1])[u,v]
                    new = new1 + new2
                weight[u,v] *= np.exp(next_model.lambda_1 * new)
            next_model.time_weight[i] = weight
        return next_model

##TEST
if __name__ == '__main__':
    print("nothing here")