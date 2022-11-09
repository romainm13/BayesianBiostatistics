import numpy as np
import matplotlib.pyplot as plt

## Functions for Data Set
def create_counts(x):
    """
    Calcul des statistiques suffisantes N pour un jeu de données x
    """
    n = x.shape[1]
    r = np.max(x) + 1
    N = {}
    # On remplit les marginales de taille 1
    for u in range(1,n+1):
        N[u] = np.zeros(r, dtype=int)
        for i in range(r):
            N[u][i] = np.sum(x[:,u-1]==i)
    # On remplit les marginales de taille 2
    for u in range(1,n+1):
        for v in range(u+1,n+1):
            N[u,v] = np.zeros((r,r), dtype=int)
            for i in range(r):
                for j in range(r):
                    N[u,v][i,j] = np.sum((x[:,u-1]==i)*(x[:,v-1]==j))
    return N

def create_counts_time(data):
    """
    On crée N un dictionnaire à t éléments qui stocke
    les statistiques suffisantes N[t] aux différents instants de mesure
    """
    N = {}
    for t, x in data.items():
        N[t] = create_counts(x)
    return N

## Functions for Model
#On tire theta oriente :theta_o PUIS on calcule theta_no
    # Orienté dict theta[u,v] pour u->v
                    #theta[u,v][i,j]=theta_vsachantu(v=j,u=i)
                    #
                    #                                   v=0   v=1
                    #   theta_uv = theta(v|u) =       |-----|-----|
                    #                             u=0 |     |     |
                    #                                 |-----|-----|
                    #                             u=1 |     |     |
                    #                                 |-----|-----|
                    #
                    #theta[u,v][i] = Dirichlet(counts[..]) donne L(v|u=i)
    # NON Orienté par formule  de Bayes

def create_theta(model):
    """
    Retourne le couple de dictionnaires (theta_o,theta_no) d'un model
    """
    n = model.n

    theta_o = {}
    for pere, les_fils in enumerate(model.tree):
        if pere == 0:
            rac = les_fils[0]
            theta_o[rac] = np.random.dirichlet((model.counts[rac][0],model.counts[rac][1]))
        if pere > 0:
            for fils in les_fils:
                if fils < pere:
                    #1) Sachant pere == 0
                    pere_0 = np.random.dirichlet((model.counts[(fils,pere)][0,0],model.counts[(fils,pere)][1,0]))
                    #2) Sachant pere == 1
                    pere_1 = np.random.dirichlet((model.counts[(fils,pere)][0,1],model.counts[(fils,pere)][1,1]))
                    theta_o[(pere,fils)] = np.array([pere_0,pere_1])
                else:
                    #1) Sachant pere == 0
                    pere_0 = np.random.dirichlet((model.counts[(pere,fils)][0,0],model.counts[(pere,fils)][0,1]))
                    #2) Sachant pere == 1
                    pere_1 = np.random.dirichlet((model.counts[(pere,fils)][1,0],model.counts[(pere,fils)][1,1]))
                    theta_o[(pere,fils)] = np.array([pere_0,pere_1])

    theta_no = {}
    # On a besoin de tous les theta_i
    for noeud in range(1,n+1):
        if noeud != model.tree[0][0]:
            theta_o[noeud] = np.random.dirichlet((model.counts[noeud][0],model.counts[noeud][1]))
            theta_no[noeud] = theta_o[noeud]
    for key,theta_vsu in theta_o.items():
        if type(key) == tuple:
            #u pere / v fils
            (u,v) = key
            if u<v:
                hg = theta_vsu[0,0]*theta_o[u][0]
                hd = theta_vsu[1,0]*theta_o[u][0]
                bg = theta_vsu[0,1]*theta_o[u][1]
                bd = theta_vsu[1,1]*theta_o[u][1]
                theta_no[key] = np.array([[hg,hd],[bg,bd]])
            if u>v:
                hg = theta_vsu[0,0]*theta_o[u][0]
                hd = theta_vsu[0,1]*theta_o[u][0]
                bg = theta_vsu[1,0]*theta_o[u][1]
                bd = theta_vsu[1,1]*theta_o[u][1]
                theta_no[(v,u)] = np.array([[hg,hd],[bg,bd]])
        else:
            theta_no[key] = theta_vsu
    return (theta_o,theta_no)
    """
    for u in range (1,n+1):
        for v in range(u+1,n+1):
                hg = theta_vsu[0,0]*theta_o[u][0]
                hd = theta_vsu[1,0]*theta_o[u][0]
                bg = theta_vsu[0,1]*theta_o[u][1]
                bd = theta_vsu[1,1]*theta_o[u][1]
                theta_no[(u,v)] = np.array([[hg,hd],[bg,bd]])
                theta_no[(v,u)] = theta_no[(u,v)]
    return (theta_o,theta_no)
    # ==> Problème : on ne peut pas calculer tous les theta_uv
    """