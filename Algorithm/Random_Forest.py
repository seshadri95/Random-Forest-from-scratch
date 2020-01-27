# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:42:58 2019

@author: sesha
"""

import random
import pandas as pd
import numpy as np
import pprint
from operator import itemgetter

# Files for this below 2 imports are attached

from Algorithm.Decision_Tree.C4_5 import * # CART implementation from scratch
from Algorithm.Utilities.Calculations import *   # All helper functions involving calculations




class random_forest():
    
    # At each tree we store features sent and their indexes which will be used for testing 
    features_idx_subseted = []
    feature_subseted = []
    
    # We store Meta info in a dictionary with assumed structure { 'Tree_1' : 'Meta_Data_frame_of_tree1',.....}
    meta_dfs = {}
    def __init__ (self,data,total_trees,bootstrap_size,feature_per_tree,tree_depth = 5,random_seed = 10):
        
        self.feature_per_tree = feature_per_tree
        self.bootstrap_size = bootstrap_size
        self.total_trees = total_trees
        self.data = data
        self.features =data.columns.tolist()[0:-1]
        self.lable_name = data.columns.tolist()[-1]
        self.tree_depth = tree_depth
        self.random_seed = random_seed
        
        
    def fit(self):
        
        '''Build each tree using Bootstrap sample Random Feature subspace.'''
        
        random.seed(self.random_seed)
               
        for tree in range(self.total_trees):
            sample = get_bootstrap(self.data,self.bootstrap_size,self.random_seed)
            
            sub_feature = random.sample(range(len(self.features)), k=self.feature_per_tree)
            random_forest.features_idx_subseted.append(sub_feature)
            
            feature_subset = list(itemgetter(*sub_feature)(self.features))
            random_forest.feature_subseted.append(feature_subset)
            
            # Call to decision tree
            
            c = DT_classifier(self.tree_depth,sample,feature_subset,self.lable_name,sample[self.lable_name].unique().tolist())
            
            c.fit()
            
            
            # Appending decision tree meta info of this tree to our random forest meta dictionary
            random_forest.meta_dfs[tree] = c.mdf


    def __predict_tree_op(self,ip_arr,cols,mdf,get_prob = 'N'):
        
        '''Predicts lables for given tree for input array of attributes.
        
        Parameters:
            Attributes Array,Meta Data frame
            
        Summary:
            -Gets all Leaf Node conditions from Meta data frame
            -Checks which condition is met by test data and maps leaf node pertaining to the condition.
            -Predicts the most probable output class the probabilities in leaf node.

        Returns:
            Predicted lables for the tree passed'''
    
        op_lst = []
        leaf_node_df = mdf [mdf['leaf_f'] == 'Y']
        leaf_conditions = leaf_node_df['split_condition']
        for elm in ip_arr:
            p = pd.Series(elm,index = cols)
            for num,condition in enumerate(leaf_conditions):
                check = []
                for cond in condition:
                    col = cond.split()[0]
                    val = ' '.join(cond.split()[1:])
                    if  eval(str(p[col]) + val):
                        check.append('True')
                    else:
                        check.append('False')
                if 'False' not in check:
                    class_prob_df = leaf_node_df.iloc[num,][self.data[self.lable_name].unique().tolist()]
                    if get_prob == 'Y':
                        op = class_prob_df.values.tolist()
                        op_lst.append(op)
                    else:
                        op = class_prob_df[class_prob_df == class_prob_df.max()].index.values #returns column name with max value
                        op_lst.append(op[0])
                    break
                        
        return np.array(op_lst)
    
    
    def predict(self,ip_arr):
        
        '''Predicts lables from output lables obtained from each tree.
        
        Parameters:
            Attributes Array,

        Returns:
            Predicted lables'''        
        
        votes_arr = []
        
        for tree in range(self.total_trees):
        
            cols = random_forest.feature_subseted[tree]
            idx_cols = random_forest.features_idx_subseted[tree]
            
            idx_data =np.array([list(arr[idx_cols]) for arr in ip_arr])
                
            op = self.__predict_tree_op(idx_data,cols,random_forest.meta_dfs[tree])
            votes_arr.append(op)
            odf = pd.DataFrame(np.array(votes_arr))
        return odf.mode().iloc[0].values
    
    def write_meta_data(self,file_name):
    
        with pd.ExcelWriter(file_name) as workbook:
            for tree,mdf in random_forest.meta_dfs.items():
                mdf.to_excel(workbook, sheet_name='Tree_'+str(tree+1),index=False)
            print("\nCheck '{}' for meta information at each tree.".format(file_name))




    
    def predict_prob(self,ip_arr):
        
        '''Predicts lables from output lables obtained from each tree.
        
        Parameters:
            Attributes Array,

        Returns:
            Predicted lables'''        
        
        votes_arr = []
        
        for tree in range(self.total_trees):
        
            cols = random_forest.feature_subseted[tree]
            idx_cols = random_forest.features_idx_subseted[tree]
            
            idx_data =np.array([list(arr[idx_cols]) for arr in ip_arr])
                
            op = self.__predict_tree_op(idx_data,cols,random_forest.meta_dfs[tree],get_prob ='Y')
            votes_arr.append(op)
        prob = np.array(votes_arr)        
        return np.mean(prob,axis=0)





