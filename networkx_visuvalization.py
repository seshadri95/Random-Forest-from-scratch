# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:04:29 2019

@author: sesha
"""
# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot
 
sheets = pd.ExcelFile('Results\\Random_Forest_Meta_Data_Generated.xlsx').sheet_names

for sheet in sheets:
    df =pd.read_excel('Results\\Random_Forest_Meta_Data_Generated.xlsx',sheet_name = sheet)
    
    nodes = df['node_number'].unique().tolist()
    
    op = {}
    op['1'] = df[df['node_number'] == '1']['split_feature'].values[0]
    for node in nodes:
        children = df[df['parent_node'] == node]['node_number']
        childs =[]
        if len(children) > 0:
            for child in children:
                split = df[df['node_number'] == child][['split_feature','split_category']].values.tolist()[0]
                x =''.join(split).split()
                rounded = str(round(float(x[-1]),3))
                x[-1] = rounded
                childs.append(' '.join(x))
                op[node] = childs
            
    l = df[df.node_number.isin(op.keys())][['split_feature','split_category']].values.tolist()
    k = [df[df['node_number'] == '1']['split_feature'].values[0]]
    k.extend([''.join(i) for i in l[1:]])

    for num,i in enumerate(k[1:]):
        x =i.split()
        rounded = str(round(float(x[-1]),3))
        x[-1] = rounded
        k[num+1] = ' '.join(x) 
    
    d = dict(zip(k,op.values()))
    
    ans = []
    for key, value in d.items():
        for i in value:
            ans.append((key,i))
    
    '''# Build your graph
    G=nx.from_dict_of_lists({1:[2,3],2:[4,5]})
     
    # Plot it
    nx.draw(G, with_labels=True)
    plt.show()'''
    
    
    import networkx as nx
    
    g=nx.DiGraph()
    g.add_edges_from(ans)
    p=nx.drawing.nx_pydot.to_pydot(g)
    
    p.write_png('Visualisation\\'+sheet+'.png')
    print('check out "Visualisation\\'+sheet+'".png for '+sheet+ ' image.')
    
    
    
