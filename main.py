import numpy as np
from bayestrees import Model

# On importe le jeu de données brutes, exemple de taille (4,3)
#   Gene     1   2   3
# cellule1   0   0   0
# cellule2   1   0   1
# cellule3   1   0   0
# cellule4   1   0   1
# Et cela pour 11 unités de temps

# Import d'un jeu de données
data = np.loadtxt('data/network3-200.txt', dtype=int, delimiter='\t')

# Instants de mesure
time = np.sort(list(set(data[:,0])))

# Dictionnaire de la forme {t: data(t)}
x = {t: data[data[:,0]==t,1:] for t in time}

# Calcul des poids pour chaque instant
n = x[0].shape[1]
model = {}
for t in time:
    model[t] = Model(n)
    model[t].update(x[t])

# Affichage
print(model[6].weight,'\n')
print(model[6].get_prob_edges())
