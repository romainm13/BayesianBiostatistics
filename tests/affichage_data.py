import sys; sys.path.append("../")
import numpy as np
from bayestrees import Model, affiche_N_V1

# Import d'un jeu de données
data = np.loadtxt('../data/network1-500.txt', dtype=int, delimiter='\t')

# Instants de mesure
time = np.sort(list(set(data[:,0])))

# Dictionnaire de la forme {t: data(t)}
x = {t: data[data[:,0]==t,1:] for t in time}

# On crée le modele
n = x[0].shape[1]

# On boucle sur les instants
for t in time:
    # Calibration du modèle
    model = Model(n)
    model.update(x[t])
    # Affichage du N a posteriori
    affiche_N_V1(model, f'N_data_{t}.pdf')
    # Affichage basique des résultats
    print(f'Instant t = {t} :')
    print(f'-> Poids {model.weight}')
    print(f'-> Proba {model.get_prob_edges()}\n')

