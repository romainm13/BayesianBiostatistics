##Libraries
import numpy as np
from scipy.special import gammaln

try:
    from .utils import create_counts, create_theta
    from .trees import random_tree, prob_edges
except:
    from utils import create_counts, create_theta
    from trees import random_tree, prob_edges

"""
Model for bayesian inference
"""

class Model:
    """
    ATTRIBUTS
    n : nombre de noeuds
    r : nombre d'états possibles
    weight : dictionnaire hyperparamètres Betas
    counts : dictionnaire hyperparamètres N_ij et N_i
    tree : arbre orienté simulé par weight ==> exemple ((1),(3),(),(2,4),())
    theta : dictionnaire des theta tirés par des Dirichelt(counts)
    """

    def __init__(self, n, r=2, s=1):
        self.n = n
        self.r = r

        #WEIGHT
        weight = {(u,v): 1 for u in range(1,n+1) for v in range(u+1,n+1)}
        self.weight = weight

        #COUNT
        #
        #   |      |      |      |
        #   N1 -- N12 -- N13 -- N14
        #          |      |      |
        #          N2 -- N23 -- N24
        #                 |      |
        #                 N3 -- N34
        #                        |
        #                        N4
        #
        #   Où
        #          1\3   3=0   3=1
        #   N13 =      |-----|-----|
        #          1=0 |  1  |  1  |
        #              |-----|-----|
        #          1=1 |  1  |  1  |
        #              |-----|-----|

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
        self.counts = counts

        #Tree based on weight
        self.tree = random_tree(self.n)

        #Oriented theta and non-oriented theta
        self.theta_o,self.theta_no = create_theta(self)

    def verif_N(self):
        """
        On verifie que les contraintes sur N sont verifiees
        N1[i] = N12[i,0] + N12[i,1] = N13[i,0] + N13[i,1] = nombre de fois que
        le sommet 1 a valu i
        N2[i] = N12[0,i] + N12[1,i] = N23[i,0] + N23[i,1] = nombre de fois que
        le sommet 2 a valu i
        N3[i] = N13[0,i] + N13[1,i] = N23[0,i] + N23[1,i] = nombre de fois que
        le sommet 3 a valu i
        """
        verif = True
        n = self.n
        N = self.counts
        Ntot = np.sum(N[1])
        for u in range(1,n+1):
            # Vérification des marginales de taille 1
            if not (np.sum(N[u]) == Ntot): verif = False
            for v in range(u+1,n+1):
                # Vérification des marginales de taille 2
                if not np.all(np.sum(N[u,v],axis=1) == N[u]): verif = False
                if not np.all(np.sum(N[u,v],axis=0) == N[v]): verif = False
        return(verif)

    def simulate(self, N, tree=None):
        """
        Simule N observations du modèle
        Parcours en profondeur
        X : observations
        pile : pile de parcours en profondeur
        pere : dictionnaire (pere,xpere)
        """
        n = self.n
        X = np.zeros((N,n), dtype=int)
        # Étape 1 : on simule l'arbre
        if tree is None:
            # Matrice des poids
            weight = np.zeros((n,n))
            for (u,v), w in self.weight.items():
                weight[u-1,v-1], weight[v-1,u-1] = w, w
            self.tree = random_tree(n, weight)
        # Étape 2 : on simule theta
        self.theta_o, self.theta_no = create_theta(self)
        # Étape 3 : on simule les observations
        for k in range(N):
            pile = [self.tree[0][0]]
            pere = dict()
            # On traite la racine
            racine = pile.pop()
            theta_racine_1 = self.theta_o[racine][1]
            x_racine = np.random.binomial(1,theta_racine_1)
            X[k,racine-1]=x_racine
            # On rajoute les fils
            for fils in self.tree[racine]:
                pile.append(fils)
                pere[fils] = (racine,x_racine)
            # On traite toute la pile
            while pile:
                noeud = pile.pop()
                # On traite le noeud
                theta = self.theta_o[pere[noeud][0],noeud][pere[noeud][1],1]
                theta_noeud_1sx_pere = theta
                x_noeud = np.random.binomial(1,theta_noeud_1sx_pere)
                X[k,noeud-1]=x_noeud
                # On rajoute les fils
                for fils in self.tree[noeud]:
                    pile.append(fils)
                    pere[fils] = (noeud,x_noeud)
        return X

    def update(self, x):
        """
        Met à jour le modèle (loi a posteriori sachant x)
        NB : x est le tableau basique des observations
        weight_uv -> weight_uv * W_uv
        N'' = N + N'

        On utilise gammaln car sinon :
        In [25]: np.log(gamma(1000))
        Out[25]: inf

        In [26]: gammaln(1000)
        Out[26]: 5905.220423209181
        """
        n, weight = self.n, self.weight
        # ATTENTION : il y a un piège ici car les dictionnaires
        # et les numpy arrays sont des objets mutables
        N = create_counts(x)
        Nold = self.counts
        Nnew = {k: Nold[k] + N[k] for k in N}
        # Mise à jour des comptages
        self.counts = Nnew
        # Mise à jour des poids
        lw = {}
        # 1. Calcul des log(Wu)
        for u in range(1,n+1):
            lw[u] = np.sum(gammaln(Nnew[u]) - gammaln(Nold[u]))
        # 2. Calcul des log(Wuv)
        for u, v in weight:
            lw[u,v] = np.sum(gammaln(Nnew[u,v]) - gammaln(Nold[u,v]))
            lw[u,v] -= lw[u] + lw[v]
        # 3. Normalisation et calcul des poids
        mlw = gammaln(np.sum(Nnew[1])) - gammaln(np.sum(Nold[1]))
        for u, v in weight:
            weight[u,v] *= np.exp(lw[u,v] + mlw)
        self.weight = weight

    def get_prob_edges(self):
        """
        Dictionnaire donnant la probabilité de chaque arrête.
        """
        n = self.n
        # Matrice des poids
        weight = np.zeros((n,n))
        for (u,v), w in self.weight.items():
            weight[u-1,v-1], weight[v-1,u-1] = w, w
        p = prob_edges(n, weight)
        return {(u,v): p[u-1,v-1] for u,v in self.weight}

##TEST
if __name__ == '__main__':
    modele1 = Model(3)
    print("weight du modele 1 :\n" , modele1.weight,"\n")
    print("counts du modele 1 :\n" , modele1.counts,"\n")
    print("theta_o du modele 1 :\n" , modele1.theta_o,"\n")
    #X = modele1.simulate(10)
    #print("10 observation sur le modele1",X,"\n")
