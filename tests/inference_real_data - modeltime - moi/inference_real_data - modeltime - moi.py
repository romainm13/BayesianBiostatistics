##Packages
import sys; sys.path.append("../../")
import os
import numpy as np
import matplotlib.pyplot as plt

from bayestrees import ModelTime
from bayestrees import create_theta, draw_graphs, get_probas_Gibbs

"""
Inference on real data using model2 = ModelTime 
cf readme.md in data
"""

# %% Parameters
network = 'network1-500.txt'
lambda_1 = 5 # weigth of similarity between previous & next time
kmax = 10 #nb of iteration in Gibbs method

# %% Importing data
network_path = '../../data/'+ network
data = np.loadtxt(network_path, dtype=int, delimiter='\t')

# Instants de mesure
list_times = np.sort(list(set(data[:,0])))
time = len(list_times)
# Dictionnaire de la forme {t: data(t)}
x = {t: data[data[:,0]==t,1:] for t in list_times}

# Useful Dicts
n = x[0].shape[1]
models = {} #Dictionnary (time, modeltime(t))
dic_p = {}  #Dictionnary (time, probabilities)

# %% Inference
model = ModelTime(n, time, lambda_1)
# MCMC Gibbs
simu_gibbs = model.sample(kmax, x)

# %%  Drawing graphs

#Creating storage folder
try:
    os.mkdir("inference_on_" + network[:-4] + f'_lambda1 = {lambda_1}')
except FileExistsError:
    pass
os.chdir("inference_on_" + network[:-4] + f'_lambda1 = {lambda_1}')

# Plot proba
tab_probas = get_probas_Gibbs(model.n, model.time, kmax, simu_gibbs)

# =============================================================================
# x = range(kmax)
# for u in range(1,n+1):
#     for v in range(u+1,n+1):
#         fig = plt.figure()
#         y = tab_probas[0][(u,v)]
#         plt.plot(x,y)
#         foo = fr"$p_{u}$" + fr"$_{v}$"
#         plt.title(foo)
#         plt.ylim(0,1)
#         plt.legend()
#         plt.savefig(f"p_{u}_{v}.png")
# =============================================================================

for t in range(model.time):
    fig = plt.figure()
    x = range(kmax)
    for u in range(1,n+1):
        for v in range(u+1,n+1):
            y = tab_probas[t][(u,v)]
            foo = fr"$p_{u}$" + fr"$_{v}$"
            plt.plot(x, y, label=foo)
            
    plt.title(f"t = {t}")
    plt.ylim(-0.1,1.1)
    plt.legend()
    plt.savefig(f"t={t}.png")
        
# %% End
os.chdir('..')