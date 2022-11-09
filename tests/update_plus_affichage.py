import sys; sys.path.append("../")
import numpy as np
from bayestrees import Model
# from bayestrees import affiche_N_V1

# Import d'un jeu de données
data = np.loadtxt('../data/network1-500.txt', dtype=int, delimiter='\t')

# Instants de mesure
time = np.sort(list(set(data[:,0])))

# Dictionnaire de la forme {t: data(t)}
x = {t: data[data[:,0]==t,1:] for t in time}

# Nombre de gènes
n = x[0].shape[1]

# Choix d'un instant de mesure
t = 7

# On crée le modele et on le met à jour
model = Model(n)
model.update(x[t])

# On enregistre N" le nouveau counts
# affiche_N_V1(model,'N.pdf')

# On observe les changements sur weight
print('Poids a posteriori :')
for (i,j), w in model.weight.items():
    print(f'{i} - {j} : {w:.2f}')
# Export des poids sous forme de matrice
weight = np.zeros((5,5))
for i,j in model.weight:
    weight[i-1,j-1] = model.weight[i,j]
    weight[j-1,i-1] = model.weight[i,j]
# weight *= 100
# Probabilités d'arrêtes dans le cas "forêts"
np.set_printoptions(precision=3, suppress=True)
l = np.diag(np.sum(weight, axis=1)) - weight
Q = np.linalg.inv(np.eye(5) + l)
print(Q)
print(np.sum(Q, axis=1))

print('Probabilités a posteriori (forêts) :')
for (i,j), w in model.weight.items():
    p = weight[i-1,j-1] * (Q[i-1,i-1] + Q[j-1,j-1] - 2*Q[i-1,j-1])
    print(f'{i} - {j} : {p:.2f}')

# Comparaison avec le cas "arbres"
print('Probabilités a posteriori (arbres) :')
for (i,j), p in model.get_prob_edges().items():
    print(f'{i} - {j} : {p:.2f}')
