# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:43:29 2022

@author: Romain Maillard
"""

import networkx as nx
import matplotlib.pyplot as plt
import random as rd

g = nx.Graph()
color_edge = []
size = []
for i in range(1,15):
    g.add_node(i)
    if i%2 == 0:
        g.add_edge(i, 4, weigth = rd.random())
        random_weight = round(rd.random(),2)*10
        color_edge.append(random_weight)
        size.append(random_weight)
        
pos = nx.circular_layout(g)

options = {  
 'font_size': 8,  
 'node_size': 200,  
 'node_color': 'white',  
 "edge_color": color_edge,
 'edgecolors': 'black',  
 'linewidths': 1,  
 'width': size,  
 'with_labels': True,  
 'verticalalignment': 'center_baseline',
 'edge_cmap' : plt.cm.Blues
 }

nx.draw(g, pos, **options)
plt.show()

# =============================================================================
# 
# # libraries
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
#  
# # Build a dataframe with your connections
# df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C'], 'value':[1, 10, 5, 5], 'width':[2,2,5,20]})
#  
# # Build your graph
# G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
# 
# pos = nx.circular_layout(G)
# 
# # Custom the nodes:
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color=[1,10,5,5],width = 5, edge_cmap=plt.cm.Blues)
# =============================================================================
