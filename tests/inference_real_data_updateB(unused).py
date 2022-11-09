##Packages
import sys; sys.path.append("../")
import os
import numpy as np
import matplotlib.pyplot as plt

from bayestrees import Model
from bayestrees import create_theta, draw_graphs

"""
Inference on real data : 
    - network0 = 3 nodes
    - network1 = 5 nodes
"""

# %% Importing data
network = 'network3-200.txt'
network_path = '../data/'+ network
data = np.loadtxt(network_path, dtype=int, delimiter='\t')

# Instants de mesure
time = np.sort(list(set(data[:,0])))

# Dictionnaire de la forme {t: data(t)}
x = {t: data[data[:,0]==t,1:] for t in time}

# Useful Dicts
n = x[0].shape[1]
models = {} #Dictionnary (time, model(t))
dic_p = {}  #Dictionnary (time, probabilities)
low_p = {}  #Dictionnary (time, low_p_uv)
threshold = 0.9 #Weight of (u,v) stronger if p_uv > threshold

# %% Inference
for t in time:
    models[t] = Model(n)
    if t > 0 :
        for edge in low_p[t]:
            models[t].weight[edge] = 0.1
    models[t].update(x[t])
    probs = models[t].get_prob_edges()
    dic_p[t] = probs
    #On stocke les arrêtes tels que les 2 noeuds sont isolés
    #On cherche les noeuds isolés
    isolated_nodes = []
    for u in range(1,n+1):
        isolated = True
        for v in range(1,n+1):
            if u < v and probs[u, v] > threshold :
                isolated = False
            if u > v and probs[v, u] > threshold :
                isolated = False
        if isolated :
            isolated_nodes.append(u)
    low_p_uv = []
    long = len(isolated_nodes)
    if long >= 2:
        for u in isolated_nodes:
            for v in isolated_nodes:
                    if u < v :
                        low_p_uv.append((u,v))
    low_p[t+1] = low_p_uv
    
# Il faut mettre à jour les thetas de chaque modeles car seuls weight et 
# counts sont MAJ
for t in time:
    models[t].theta_o,models[t].theta_no = create_theta(models[t])

# %% 1stHELP : Drawing graphs
#Creating storage folder
try:
    os.mkdir(f"inference_updateB_s={threshold}_" + network[:-4])
except FileExistsError:
    pass
os.chdir(f"inference_updateB_s={threshold}_" + network[:-4])
draw_graphs(n,dic_p,threshold)

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
plt.show()

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
plt.show()

# %% End
os.chdir('..')