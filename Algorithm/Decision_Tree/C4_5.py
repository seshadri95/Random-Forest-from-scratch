# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:30:13 2019

@author: sesha
"""
import pandas as pd
import numpy as np

from Algorithm.Utilities.Calculations import *


class DT_classifier():
    
    # Meta Data Frame to store decision tree information
    mdf  = pd.DataFrame()
    
    def __init__(self,max_depth,train_df,feature_list,lable_name,lable_list):
        self.max_depth = max_depth
        self.train_df = train_df
        self.feature_list = feature_list
        self.lable_name = lable_name
        self.lable_list = lable_list

    
    def __tree_initial_split(self,feature_list,lable,df_test):
        
        """Initiate initial tree split to get data into meta df for further spliting.
        
        Parameters -
            
            Attributes,Label name,Data

        Returns -
            
            Meta Data Frame
        """
        split_feature,threshold,overall_entropy = discretize(df_test,lable,feature_list)
        cols = ['node_number','parent_node','split_feature','split_category','split_condition','overall_entropy','leaf_f','next_split_features']
        val = [['1',np.nan,split_feature,np.nan,[threshold],overall_entropy,'N']]
        feature_for_next_split = feature_list.copy()
        feature_for_next_split.remove(split_feature)  # removing current split feature for further spliting
        val[0].append(feature_for_next_split)    
        lables = df_test[lable].unique().tolist()
        for i in lables:
            cols.append(str(i))   # creating columns as per no of labels in op class
            class_data = len(df_test[df_test[lable] == i])
            total_data = len(df_test)
            val[0].append(class_data/total_data) # probability for that lable in the node
        meta_df = pd.DataFrame(val,columns=cols)
        #for num,levels in enumerate(df_test[split_feature].unique()): #Branches from the first node
        for num,operator in enumerate([' <= ',' > ']):
            condition = operator +str(threshold)
            vals = [['1_'+str(num+1),'1',split_feature,condition,[split_feature+condition],np.nan,np.nan,feature_for_next_split]]
            for lable in lables: vals[0].append(np.nan) #appending values for op class probability
            meta_df = meta_df.append(pd.DataFrame(vals,columns=cols),ignore_index=True)[cols] # adding observations to meta_df
        return meta_df
        
        
        
    
    
    def __build_tree(self,data,meta_df,lable_name,lable_list,final_split = 'N'):
        
        '''Split tree from the provided state to next state.
        
        Parameters:
            
            Data,Meta Data Frame,Label name,Classes in lable,Final split flag
        
        Summary:
            
            -Filters the nodes that are not split using meta info
            -Filtered Nodes which are non-terminal will be split further.
            -Filtered Nodes are updated with Leaf flag,overall entropy and class probability information in meta data frame.
            -If final split flag = 'Y', only weights are updated no further spliting (If tree_depth has reached but node is not leaf then we do this as last step)
            
        Returns:
            Temporary Data Frame containg meta information of new split done'''
        
        df_node_to_split = meta_df[(meta_df['parent_node'].notnull()) & (pd.isnull(meta_df['overall_entropy'])) ]
        cols = meta_df.columns.tolist()
        temp_df = pd.DataFrame(columns=cols)
        for i in range(0,len(df_node_to_split)):
            leaf = False
            series = df_node_to_split.iloc[i,]
            node_number,features_for_split,split_condition = series['node_number'],series['next_split_features'],series['split_condition']
            
            #Filtering data subset based on split condition dict
            subset = data
            for cond in split_condition:
                key = cond.split()[0]
                value = ' '.join(cond.split()[1:])
                subset = eval('subset[subset[key]'+ value+']')
            
            probs = []
            #Updating class label  propbailites and lead flag
            for lable in lable_list:
                if len(subset) >0: # for zreo division error
                    prob = len(subset[subset[lable_name] == lable]) / len(subset)
                    meta_df.loc[meta_df['node_number'] ==series['node_number'],str(lable)] = prob
                    probs.append(prob)
            if (1 in probs) or (final_split == 'Y') or (len(features_for_split) == 0):
                meta_df.loc[meta_df['node_number'] ==series['node_number'],'leaf_f'] = 'Y'
                meta_df.loc[meta_df['node_number'] ==series['node_number'],'overall_entropy'] = 0
                leaf = True
            else:
                meta_df.loc[meta_df['node_number'] ==series['node_number'],'leaf_f'] = 'N'
                    
            if leaf or (final_split == 'Y') :
                continue
            
            #Get split feature for further split if its not a leaf node
            
            split_feature,threshold,overall_entropy = discretize(subset,lable_name,features_for_split)
            meta_df.loc[meta_df['node_number'] ==series['node_number'],'overall_entropy'] = overall_entropy #Weighted entropy update
            next_features = features_for_split.copy()
            next_features.remove(split_feature) # child's next feature for split
            
            #child observations entry
            for num,operator in enumerate([' <= ',' > ']):
            #for num,levels in enumerate(subset[split_feature].unique()):
                condition = split_condition.copy()
                condition.append(split_feature + operator + str(threshold))
                vals = [[node_number+'_'+str(num+1),node_number,split_feature,operator + str(threshold),condition,np.nan,np.nan,next_features]]
                for lbl in lable_list: vals[0].append(np.nan)              
                temp_df = temp_df.append(pd.DataFrame(vals,columns=cols),ignore_index=True)[cols]
        return temp_df
        
    def __recursive_tree(self,data,max_depth,lable_name,feature_list,lable_list):

        '''Initiates tree building and splits child nodes until convergence.
        
        Parameters:
            Data,Tree Depth,Label name,Data,Attributes,Output class names

        Returns:
            Meta Data Frame'''
            
        meta_df = self.__tree_initial_split (feature_list,lable_name,data)
        max_depth = max_depth - 1 #as already 1 split has been done by us
        for i in range(0,max_depth): 
            temp_df = self.__build_tree(data,meta_df,lable_name,lable_list)
            meta_df = meta_df.append(temp_df,ignore_index=True)[meta_df.columns.tolist()]
        temp_df = self.__build_tree(data,meta_df,lable_name,lable_list,final_split = 'Y') # just to update meta data no split occurs here
        meta_df = meta_df.append(temp_df,ignore_index=True)[meta_df.columns.tolist()]
        return meta_df        
        

       
        
    def fit(self):
        df = self.__recursive_tree(self.train_df,self.max_depth,self.lable_name,self.feature_list,self.lable_list)
        DT_classifier.mdf = df
        
    
    


