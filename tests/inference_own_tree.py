# %% Packages
import sys; sys.path.append("../")
import os
import numpy as np
import matplotlib.pyplot as plt

from bayestrees import Model
from bayestrees import create_theta
from bayestrees import affiche_N_V1, affiche_N_V2, affiche_theta_no_V1

"""
First try with
-> n = 5 nodes
-> weight 10 if edge and 1 if not
-> counts 11/1 (for positive correlation), 6 (equal for all if no correlation)

Only positive correlations
"""
# %% Creating the tree we want to infere after
n = 5
model = Model(n)

#Structure : tree
model.tree = ((1,),(2,5),(3,4),(),(),())

#Structure : weight
list = [(1,2),(1,5),(2,3),(2,4)]
for couple in list:
    model.weight[couple] = 10

#Correlation : counts
#The more Nij is low, the more the distibution of theta is asymmetric
#N'' = N' + N so the weight between N' and N must be balanced
for u in range(1,n+1):
    for v in range(u+1,n+1):
        if (u,v) in list:
            model.counts[u,v] = np.array([[11,1],[1,11]])
        else:
            model.counts[u,v] = 6*np.ones((2,2), dtype = np.int32)
for u in range(1,n+1):
    model.counts[u] = np.zeros(2)
    if u < n:
        model.counts[u] = np.sum(model.counts[(u,u+1)], axis=1)
    else:
        model.counts[u] = np.sum(model.counts[(1,u)], axis=0)
print('Counts well defined ? :',model.verif_N(),'\n')

#Theta: depends on counts
model.theta_o,model.theta_no = create_theta(model)
print('Oriented theta:\n', model.theta_o,'\n')

# %% Creating storage folder
try:
    os.mkdir('inference_own_tree')
except FileExistsError:
    pass
os.chdir('inference_own_tree')
#affiche_N_V1(model,'les_N_init_V1.pdf')
#affiche_N_V2(model,'les_N_init_V2.pdf')
affiche_theta_no_V1(model,'les_theta_init_V1.pdf')

# %% Simulating our tree to obtain datas
data_len = 500
np.random.seed(1)
#On a un jeu de données stable (theta "assez équilibré")
data = model.simulate(data_len,model.tree)
np.random.seed(5)
np.random.shuffle(data)
print('Data calculated by the initial model:\n', data,'\n')


# %% Trying to infere with data
new_model = Model(n)
#affiche_N_V2(new_model,'les_N_new_model_empty_V2.pdf')

new_model.update(data)
#affiche_N_V2(new_model,'les_N_new_model_updated_V2.pdf')

print('Weight initial model:',model.weight,'\n')
print('Probability initial model',model.get_prob_edges(),'\n')
print('Weight inffered model:',new_model.weight,'\n')
print('Probability inffered model',new_model.get_prob_edges(),'\n')

# %% Calculation of probabilities depending on the number of data used
p = {} #Dictionnary of probabilities of edges

for u in range(1,n+1):
    for v in range(u+1,n+1):
        p[u,v] = []

#Adding a new data one by one and calculating probabilityes of the edges
for nb_data in range(1,data_len+1):
    _data = data[:nb_data]
    new_model = Model(n)
    new_model.update(_data)
    probs = new_model.get_prob_edges()
    for u in range(1,n+1):
        for v in range(u+1,n+1):
            p[u,v].append(probs[u,v])


#affiche_N(model,"N_data_initial.pdf")
#affiche_N(new_model,"N_data_inffered.pdf")

# %% Plot probabilities
X = np.arange(data_len) + 1

for u in range(1,n+1):
    for v in range(u+1,n+1):
        Y = p[u,v]
        plt.plot(X, Y, label=f"p_{u}{v}")

plt.title("Probs = f(nb_data) - n = 5; B = 10/1; Counts = 11/1 or 6")
plt.ylim(0,2)
plt.legend()
plt.show()

os.chdir('..')