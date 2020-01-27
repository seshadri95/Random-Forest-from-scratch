# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:32:24 2019

@author: sesha
"""

import numpy as np

def get_threshold(df,col,lable):
    
    '''Get best threshold value for the attribute
        
        Parameters : 
            Data frame,Attribute,Lable name
        
        Returns:
            Entropy value,Threshold value'''
    
    df = df[[col,lable]]
    uniq_vals = df[col].unique().tolist()
    uniq_vals.sort()
    thresholds = [(uniq_vals[idx]+uniq_vals[idx+1])/2 for idx in range(0,len(uniq_vals) -1)]
    weighted_entropy = []
    
    for threshold in thresholds:
        left = df[df[col] <= threshold][lable]
        right = df[df[col] > threshold][lable]
        entropy = calc_wgtd_entropy_numeric(left,right)
        weighted_entropy.append(entropy)
    if thresholds == []:
        weighted_entropy = [0]
        thresholds = [0]
        
    return [min(weighted_entropy),thresholds[weighted_entropy.index(min(weighted_entropy))]]
        
        

def calc_wgtd_entropy_numeric( left, right):
    
    '''Get overall entropy for an attribute
        
        Parameters : 
            Data towards left and right of threshold (i.e, <= threshold & > threshold )
            
        Returns:
            Overall entropy'''
            
    total_elements = len(left) + len(right)
    ent_left = entropy(left)
    ent_right = entropy(right)
    weighted_entropy = ((len(left) / total_elements) * ent_left) + ((len(right) / total_elements) * ent_right)
    return weighted_entropy


def entropy(df):
     
    '''Get entropy for a subset
        
        Parameters : 
            Lable values of subset
            
        Returns:
            Entropy'''
            
    op_class, count = np.unique(df.values, return_counts=True)
    entropy = np.sum([(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count)) for i in range(len(op_class))])
    return entropy

    
    
def discretize(df,lable,feature_list):
    
    '''Discretization (binning) of continuous data
        
        Parameters : 
            Data Frame,Lable Name,Attributes
            
        Returns:
            Attribute,Threshold,Overall Entropy'''    
    
    val = []
    d = {}
    for i in feature_list:
        op = get_threshold(df,i,lable)
        val.append(op)
        d[op[1]] = i

    ent = [value[0] for value in val]
    thr = [value[1] for value in val]
    return d[thr[ent.index(min(ent))]],thr[ent.index(min(ent))],min(ent)


def get_bootstrap(data, bootstrap_size,random_seed):
    
    ''' Bootstraping for random forrest
        
        Parameters : 
            Data Frame, Bootstarp size, Random state
            
        Returns:
            Boostrap sample'''
            
    np.random.seed(random_seed)
    bootstraps = np.random.randint(low=0, high=len(data), size=bootstrap_size)
    df_bootstrap = data.iloc[bootstraps]
    return df_bootstrap