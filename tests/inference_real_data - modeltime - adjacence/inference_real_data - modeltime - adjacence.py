##Packages
import sys; sys.path.append("../../")
import os
import numpy as np
import matplotlib.pyplot as plt

from bayestrees import ModelTimeAdj, Model
from bayestrees import draw_graphs_adj, get_probas_Gibbs, probs_
from bayestrees import prob_edges_weight, draw_graphs

"""
Inference on real data using model2 = ModelTime 
cf readme.md in data
"""

# %% Parameters
network = 'network0-200.txt'
lambda_1 = 0 # weigth of similarity between previous & next time
kmax = 1000 # number of iterations in Gibbs method
# np.random.seed(0)

# %% Importing data
network_path = '../../data/'+ network
data = np.loadtxt(network_path, dtype=int, delimiter='\t')

# Instants de mesure
list_times = np.sort(list(set(data[:,0])))
time = len(list_times)
# Dictionnaire de la forme {t: data(t)}
x = {t: data[data[:,0]==t,1:] for t in list_times}
n = x[0].shape[1]

# %% INFERENCE
# Inference former model
models = {} #Dictionnary (time, model(t))
dic_p = {}  #Dictionnary (time, probabilities)
for t in range(time):
    models[t] = Model(n)
    models[t].update(x[t])
    dic_p[t] = models[t].get_prob_edges()
    
# Inference ModelTimeAdj
model = ModelTimeAdj(n, time, lambda_1)
# model.time_weight *= 1e-5
# MCMC Gibbs
simu_gibbs = model.sample(kmax, x)

# %% Creating storage folder
try:
    os.mkdir("inference_on_" + network[:-4] + f'_lambda1 = {lambda_1}')
except FileExistsError:
    pass
os.chdir("inference_on_" + network[:-4] + f'_lambda1 = {lambda_1}')

# %%
## 1st WAY TO VIZUALIZE : PLOT PROBAS
probs_(simu_gibbs, dic_p)

# =============================================================================
# ## 2nd WAY TO VIZUALIZE : DRAW GRAPHS WITH ADJ MATRIX
# ResultGibbs = simu_gibbs[-1].time_adj
# # For each of the {time} models we draw the graph
# draw_graphs_adj(n,ResultGibbs)
# =============================================================================

# =============================================================================
# ## 3rd WAY TO VIZUALIZE : DRAW GRAPHS WITH probas of edges
# ResultGibbs = simu_gibbs[-1].time_weight
# dic_p = {}
# for t in range(time):
#     dic_p[t] = prob_edges_weight(n, ResultGibbs[t])
# draw_graphs(n,dic_p)
# =============================================================================

        
# %% End
os.chdir('..')
