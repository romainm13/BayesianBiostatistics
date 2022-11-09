##Packages
import sys; sys.path.append("../../")
import os
import numpy as np
import matplotlib.pyplot as plt

from bayestrees import Model
from bayestrees import create_theta, draw_graphs, draw_graphs_5nodes

"""
Inference on real data : 
cf readme.md in data
"""
# %% SEED
np.random.seed(0)

# %% Importing data
network = 'network5-500.txt'
network_path = '../../data/'+ network
data = np.loadtxt(network_path, dtype=int, delimiter='\t')

# Instants de mesure
time = np.sort(list(set(data[:,0])))

# Dictionnaire de la forme {t: data(t)}
x = {t: data[data[:,0]==t,1:] for t in time}

# Useful Dicts
n = x[0].shape[1]
models = {} #Dictionnary (time, model(t))
dic_p = {}  #Dictionnary (time, probabilities)

# %% Inference
for t in time:
    models[t] = Model(n)
    models[t].update(x[t])
    dic_p[t] = models[t].get_prob_edges()

# Il faut mettre à jour les thetas de chaque modeles car seuls weight et 
# counts sont MAJ
for t in time:
    models[t].theta_o,models[t].theta_no = create_theta(models[t])
# %% 1stHELP : Drawing graphs
#Creating storage folder
try:
    os.mkdir("inference_on_" + network[:-4])
except FileExistsError:
    pass
os.chdir("inference_on_" + network[:-4])
draw_graphs(n,dic_p) # General function
#draw_graphs_5nodes(n,dic_p) # function for the report
# %% 2ndHELP : Drawing theta_u ON AVERAGE to avoid variation from Dirichlet
fig = plt.figure()
for u in range(1,n+1):
    theta_u_1_mean = []
    for t in time:
            theta_u_1_mean.append(models[t].counts[u][1]
                                  /np.sum(models[t].counts[u]))
    foo = fr"$\theta_{u}$" + r"$^{mean}$(1)" 
    plt.plot(time, theta_u_1_mean, label=foo)

plt.title(r"Évolution des $\theta_u^{mean}(1)$")
plt.ylim(0,1.5)
plt.legend()
plt.savefig("ev_theta_u(1)_mean.png")

# %% 3rdHELP : Drawing pearson_uv ON AVERAGE to avoid variation from Dirichlet
# and to avoid calculate all the theta_uv
fig = plt.figure()
for u in range(1,n+1):
    for v in range(u+1,n+1):
        p_uv_mean = [] #store p_uv at each time
        for t in time:
            N = np.sum(models[t].counts[1])
            p_uv_t = ((N * models[t].counts[u,v][1,1] 
                      - models[t].counts[u][1] * models[t].counts[v][1])
                      / np.sqrt(models[t].counts[u][0] 
                                * models[t].counts[u][1]
                                * models[t].counts[v][0]
                                * models[t].counts[v][1]
                                ))
            p_uv_mean.append(p_uv_t)
        foo = fr"$\rho_{u}$" + fr"$_{v}$" + r"$^{mean}$"
        plt.plot(time, p_uv_mean, label=foo)

plt.title(r"Évolution des $\rho_{uv}^{mean}$")
plt.xlim(-0.1,max(time) + 3)
plt.ylim(-0.5,0.7)
plt.legend()
plt.savefig("ev_p_uv_mean.png")

# %% End
os.chdir('..')
