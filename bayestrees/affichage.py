import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

def affiche_N(model, fname, var=None, cmap=None):
    """
    Enregistre dans un .pdf/.png "fname" la matrice N normalisée par Ntot
    """
    n = model.n
    Ntot = np.sum(model.counts[1])
    # Paramètres graphiques
    scale = 1.5
    style = {'lw': 0.5, 'c': 'k'}
    if var is None: var = 'N'
    if cmap is None: cmap = plt.get_cmap('Oranges')
    # Figure
    fig = plt.figure(figsize=(scale*(n+1),scale*(n+1)))
    gs = gridspec.GridSpec(n+1, n+1, hspace=0.4,
        height_ratios=[0.5]+n*[1], width_ratios=[0.5]+n*[1])
    # Marginales de taille 1 : version horizontale
    for u in range(1,n+1):
        ax = plt.subplot(gs[0,u])
        ax.set_aspect('equal')
        ax.set_title(f'${var}_{{{u}}}$')
        ax.plot([0.5,0.5], [0.5,1], **style)
        for i in [0,1]:
            p = model.counts[u][i]/Ntot
            rect = Rectangle((i/2,0.5), 0.5, 0.5, color=cmap(p), lw=0)
            ax.add_patch(rect)
        ax.set(xlim=(0,1), ylim=(0.5,1))
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', left=False, labelleft=False)
    # Marginales de taille 1 : version verticale
    for u in range(1,n+1):
        ax = plt.subplot(gs[u,0])
        ax.set_aspect('equal')
        ax.set_title(f'${var}_{{{u}}}$')
        ax.plot([0,0.5], [0.5,0.5], **style)
        for i in [0,1]:
            p = model.counts[u][i]/Ntot
            rect = Rectangle((0,0.5-i/2), 0.5, 0.5, color=cmap(p), lw=0)
            ax.add_patch(rect)
        ax.set(xlim=(0,0.5), ylim=(0,1))
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', left=False, labelleft=False)
    # Marginales de taille 2 : triangle supérieur
    for u in range(1,n):
        for v in range(u+1,n+1):
            # On sélectionne l'élément de la grille
            ax = plt.subplot(gs[u,v])
            ax.set_aspect('equal')
            # Afficher le quadrillage
            ax.set(xlim=(0,1), ylim=(0,1))
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', left=False, labelleft=False)
            # On remplit le tableau
            ax.set_title(f'${var}_{{{u}{v}}}$')
            ax.plot([0,1], [0.5,0.5], **style)
            ax.plot([0.5,0.5], [0,1], **style)
            for i, j in [(0,0),(0,1),(1,0),(1,1)]:
                p = model.counts[(u,v)][i,j]/Ntot
                rect = Rectangle((j/2,(1-i)/2), 0.5, 0.5,
                    color=cmap(p), lw=0)
                ax.add_patch(rect)
    # Marginales de taille 2 : triangle inférieur
    for u in range(1,n):
        for v in range(u+1,n+1):
            # On sélectionne l'élément de la grille
            ax = plt.subplot(gs[v,u])
            ax.set_aspect('equal')
            # Afficher le quadrillage
            ax.set(xlim=(0,1), ylim=(0,1))
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', left=False, labelleft=False)
            # On remplit le tableau
            ax.set_title(f'${var}_{{{v}{u}}}$')
            ax.plot([0,1], [0.5,0.5], **style)
            ax.plot([0.5,0.5], [0,1], **style)
            for i, j in [(0,0),(0,1),(1,0),(1,1)]:
                p = model.counts[(u,v)][j,i]/Ntot
                rect = Rectangle((j/2,(1-i)/2), 0.5, 0.5,
                    color=cmap(p), lw=0)
                ax.add_patch(rect)
    # Enregistrement de la figure
    fig.savefig(fname, bbox_inches='tight')

def affiche_theta_no(model, fname):
    """
    Enregistre dans "fname" la matrice theta (non orientée)
    """
    var = r'\theta'
    cmap = plt.get_cmap('Blues')
    affiche_N(model, fname, var=var, cmap=cmap)


# BACKUP versions précédentes
def affiche_N_V1(model, fname):
    cmap = plt.get_cmap('Oranges')
    """
    Enregistre dans un .pdf/.png "fname" la matrice N normalisée par Ntot
    """
    n = model.n
    Ntot = np.sum(model.counts[1])
    # print(f'Ntot = {Ntot}')
    # Figure
    fig = plt.figure(figsize=(2*n,2*n))
    gs = gridspec.GridSpec(n,n, hspace=0.4)
    # Boucle sur les arrêtes
    for u in range(1,n+1):
        for v in range(u,n+1):
            # On sélectionne l'élément de la grille
            ax = plt.subplot(gs[u-1,v-1])
            ax.set_aspect('equal')
            # Afficher le quadrillage
            ax.set(xlim=(0,1), ylim=(0,1))
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', left=False, labelleft=False)
            # On remplit le tableau
            if u == v:
                ax.axis('off')
                ax.set_title(r'$N_{' + f'{u}' + r'}$')
                ax.plot([0,0,1,1], [0.51,1,1,0.51], lw=1.8, c='k')
                ax.plot([0,1], [0.5,0.5], lw=0.8, c='k')
                ax.plot([0.5,0.5], [0.5,1], lw=0.5, c='k')
                for i, j in [(0,1),(1,1)]:
                    p = model.counts[u][i]/Ntot
                    rect = Rectangle((i/2,j/2), 0.5, 0.5, color=cmap(p), lw=0)
                    ax.add_patch(rect)
            else:
                ax.set_title(r'$N_{' + f'{u}{v}' + r'}$')
                ax.plot([0,1], [0.5,0.5], lw=0.5, c='k')
                ax.plot([0.5,0.5], [0,1], lw=0.5, c='k')
                for i, j in [(0,0),(0,1),(1,0),(1,1)]:
                    p = model.counts[(u,v)][i,j]/Ntot
                    rect = Rectangle((j/2,(1-i)/2), 0.5, 0.5,
                        color=cmap(p), lw=0)
                    ax.add_patch(rect)
    # Enregistrement de la figure
    fig.savefig(fname, bbox_inches='tight')

def affiche_theta_no_V1(model, fname):
    cmap = plt.get_cmap('Blues')
    """
    Enregistre dans un .png "fname" la matrice N
    """
    n = model.n
    # Figure
    fig = plt.figure(figsize=(2*n,2*n))
    gs = gridspec.GridSpec(n,n, hspace=0.4)
    # Boucle sur les arrêtes
    for u in range(1,n+1):
        for v in range(u,n+1):
            # On sélectionne l'élément de la grille
            ax = plt.subplot(gs[u-1,v-1])
            ax.set_aspect('equal')
            # Option afficher le quadrillage
            ax.plot([0,1], [0.5,0.5], lw=0.5, c='k')
            ax.plot([0.5,0.5], [0,1], lw=0.5, c='k')
            ax.set(xlim=(0,1), ylim=(0,1))
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', left=False, labelleft=False)
            title = r'$\theta_{' + f'{u}{v}' + r'}$'
            if u == v:
                title = r'$\theta_{' + f'{u}' + r'}$'
            ax.set_title(title)
            # On remplit le tableau
            if u == v:
                for i in [0,1]:
                    p = model.theta_no[u][i]
                    rect = Rectangle((i/2,(1-i)/2), 0.5, 0.5, color=cmap(p), lw=0)
                    ax.add_patch(rect)
            elif (u,v) in model.theta_no:
                for i, j in [(0,0),(0,1),(1,0),(1,1)]:
                    p = model.theta_no[(u,v)][i,j]
                    if (i,j) == (0,0):
                        rect = Rectangle((0,1/2), 0.5, 0.5, color=cmap(p), lw=0)
                        ax.add_patch(rect)
                    if (i,j) == (0,1):
                        rect = Rectangle((1/2,1/2), 0.5, 0.5, color=cmap(p), lw=0)
                        ax.add_patch(rect)
                    if (i,j) == (1,0):
                        rect = Rectangle((0,0), 0.5, 0.5, color=cmap(p), lw=0)
                        ax.add_patch(rect)
                    if (i,j) == (1,1):
                        rect = Rectangle((1/2,0), 0.5, 0.5, color=cmap(p), lw=0)
                        ax.add_patch(rect)
    # Enregistrement de la figure
    fig.savefig(fname, bbox_inches='tight')

def affiche_N_V2(model, fname, var=None, cmap=None):
    """
    Enregistre dans un .pdf/.png "fname" la matrice N normalisée par Ntot
    """
    n = model.n
    Ntot = np.sum(model.counts[1])
    # Paramètres graphiques
    hr = [0.5] + (n-1)*[1]
    wr = (n-1)*[1] + [0.5]
    scale = 1.5
    style = {'lw': 0.5, 'c': 'k'}
    if var is None: var = 'N'
    if cmap is None: cmap = plt.get_cmap('Oranges')
    # Figure
    fig = plt.figure(figsize=(scale*n,scale*n))
    gs = gridspec.GridSpec(n, n, hspace=0.4, height_ratios=hr, width_ratios=wr)
    # Marginales de taille 1 : version horizontale
    for u in range(2,n+1):
        ax = plt.subplot(gs[0,u-2])
        ax.set_aspect('equal')
        ax.set_title(f'${var}_{{{u}}}$')
        ax.plot([0.5,0.5], [0.5,1], **style)
        for i in [0,1]:
            p = model.counts[u][i]/Ntot
            rect = Rectangle((i/2,0.5), 0.5, 0.5, color=cmap(p), lw=0)
            ax.add_patch(rect)
        ax.set(xlim=(0,1), ylim=(0.5,1))
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', left=False, labelleft=False)
    # Marginales de taille 1 : version verticale
    for u in range(1,n):
        ax = plt.subplot(gs[u,n-1])
        ax.set_aspect('equal')
        ax.set_title(f'${var}_{{{u}}}$')
        ax.plot([0,0.5], [0.5,0.5], **style)
        for i in [0,1]:
            p = model.counts[u][i]/Ntot
            rect = Rectangle((0,0.5-i/2), 0.5, 0.5, color=cmap(p), lw=0)
            ax.add_patch(rect)
        ax.set(xlim=(0,0.5), ylim=(0,1))
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', left=False, labelleft=False)
    # Marginales de taille 2
    for u in range(1,n):
        for v in range(u+1,n+1):
            # On sélectionne l'élément de la grille
            ax = plt.subplot(gs[u,v-2])
            ax.set_aspect('equal')
            # Afficher le quadrillage
            ax.set(xlim=(0,1), ylim=(0,1))
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', left=False, labelleft=False)
            # On remplit le tableau
            ax.set_title(f'${var}_{{{u}{v}}}$')
            ax.plot([0,1], [0.5,0.5], **style)
            ax.plot([0.5,0.5], [0,1], **style)
            for i, j in [(0,0),(0,1),(1,0),(1,1)]:
                p = model.counts[(u,v)][i,j]/Ntot
                rect = Rectangle((j/2,(1-i)/2), 0.5, 0.5,
                    color=cmap(p), lw=0)
                ax.add_patch(rect)
    # Enregistrement de la figure
    fig.savefig(fname, bbox_inches='tight')

# %% ModelTimeAdj
## Drawing graphs with networkx
def draw_graphs(n_nodes,dic_p):
    """
    Input :
    n : nb of nodes
    dic_p :Dictionary (time, dic_p[t] dictionary with probs of (u,v))
    Output : t files.png of the graphs keeping edges with p_uv > threshold
    """
    fig = plt.figure()
    
    g = nx.Graph()
    color_edge = []
    sizes_edge = []
    for t,p in dic_p.items():
        for u in range (1,n_nodes+1):
            g.add_node(u)
            for v in range (u+1,n_nodes+1):
                    g.add_edge(u,v)
                    color_edge.append(p[u,v]*10)
                    sizes_edge.append(p[u,v]*10)
        #Drawing
        pos = nx.circular_layout(g, dim=2, scale=1)
        options = {  
         'font_size': 8,  
         'node_size': 200,
         'node_color': 'white',  
         "edge_color": color_edge,
         'edgecolors': 'black',  
         'linewidths': 1,  
         'width': sizes_edge,  
         'with_labels': True,  
         'verticalalignment': 'center_baseline',
         'edge_cmap' : plt.cm.Blues
         }
        nx.draw_networkx(g, pos, **options)
        fig.savefig(f"graph_t={t}.png")
        plt.clf()
        color_edge = []
        sizes_edge = []

def draw_graphs_5nodes(n_nodes,dic_p):
    """
    Input :
    n : 5 nb of nodes
    dic_p :Dictionary (time, dic_p[t] dictionary with probs of (u,v))
    Output : t files.png of the graphs keeping edges with p_uv > threshold
    """
    fig = plt.figure()
    
    g = nx.Graph()
    color_edge = []
    sizes_edge = []
    for t,p in dic_p.items():
        for u in range (1,n_nodes+1):
            g.add_node(u)
            for v in range (u+1,n_nodes+1):
                    g.add_edge(u,v)
                    color_edge.append(p[u,v]*10)
                    sizes_edge.append(p[u,v]*10)
        #Drawing
        pos = pos = {1: (100, 100), 2: (95, 95), 3: (105, 95), 4: (95, 90), 5: (105, 90)} 
        options = {  
         'font_size': 8,  
         'node_size': 200,
         'node_color': 'white',  
         "edge_color": color_edge,
         'edgecolors': 'black',  
         'linewidths': 1,  
         'width': sizes_edge,  
         'with_labels': True,  
         'verticalalignment': 'center_baseline',
         'edge_cmap' : plt.cm.Blues
         }
        nx.draw_networkx(g, pos, **options)
        fig.savefig(f"graph_t={t}.png")
        plt.clf()
        color_edge = []
        sizes_edge = []
        
        
def draw_graphs_adj(n_nodes,dic_adj):
    """
    Input :
    n : nb of nodes
    dic_adj :Dictionary (time, dic_adj[t] dictionary of adj matrix)
    Output : t files.png of the graphs keeping edges with p_uv > threshold
    """
    fig = plt.figure()
    
    g = nx.Graph()
    color_edge = []
    sizes_edge = []
    for t,adj in dic_adj.items():
        for u in range (1,n_nodes+1):
            g.add_node(u)
            for v in range (u+1,n_nodes+1):
                    g.add_edge(u,v)
                    color_edge.append(adj[u-1,v-1]*10)
                    sizes_edge.append(adj[u-1,v-1]*10)
        #Drawing
        pos = nx.circular_layout(g, dim=2, scale=1)
        options = {  
         'font_size': 8,  
         'node_size': 200,
         'node_color': 'white',  
         "edge_color": color_edge,
         'edgecolors': 'black',  
         'linewidths': 1,  
         'width': sizes_edge,  
         'with_labels': True,  
         'verticalalignment': 'center_baseline',
         'edge_cmap' : plt.cm.Blues
         }
        nx.draw_networkx(g, pos, **options)
        fig.savefig(f"graph_t={t}.png")
        plt.clf()
        color_edge = []
        sizes_edge = []
        
def probs_(simu_gibbs, dic_p):
    """
    Parameters
    ----------
    [Z^(0), Z^(1), Z^(2), Z^(3) ..., Z^(kmax-1)] : 
    List of models of all the states of MCMC
    
    Returns
    -------
    Nothing but save figs
    """
    kmax, time, n = simu_gibbs.shape[:3]
    for t in range(time):

        fig = plt.figure()
        x = np.arange(1, kmax+1)
        mean_p = np.cumsum(simu_gibbs, axis=0)
        n_iter = np.cumsum(np.ones(kmax))

        for u in range(1,n+1):
            for v in range(u+1,n+1):
                label = fr'$\hat{{p}}_{{{u}{v}}}$'
                plt.plot(x, mean_p[:,t,u-1,v-1]/n_iter, label=label)
                plt.axhline(y=dic_p[t][u,v], color='r')
                plt.text(0, dic_p[t][u,v]+.02, f'{u}{v} plim={dic_p[t][u,v]}')
    
        plt.title(f"t = {t}")
        plt.xlim(0,np.max(x))
        plt.ylim(0,1)
        plt.legend()
        fig.savefig(f"t={t}.png")
