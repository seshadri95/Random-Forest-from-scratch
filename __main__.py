import random
import pandas as pd
import numpy as np
from Algorithm.Random_Forest import random_forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score,cross_val_predict

df = pd.read_csv('Data\\hazzlenuts_preprocessed.csv')

features = df.columns[0:-1].tolist()
output_lable_name = df.columns[-1]


x = df[features].values
y = df[output_lable_name].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 1)

train_df = pd.DataFrame(x_train,columns = features)
train_df[output_lable_name] = y_train
train_df

def gen_random_splits(n):
    sk_rf_acc = []
    rf_acc = []
    for split in range(0,n):
        
        print('\n---------------------------------Random split - {} -------------------'.format(split+1))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = split)
        
        train_df = pd.DataFrame(x_train,columns = features)
        train_df[output_lable_name] = y_train
        train_df
        
        r = random_forest(data = train_df,total_trees = 11,tree_depth = 5,random_seed = 10,feature_per_tree = 6,bootstrap_size=100)
        r.fit()
        op= r.predict(x_test)
        acc_rf = metrics.accuracy_score(y_test, op)
        print('\nAccuracy for random_forest (implemented)= {}%'.format(round(acc_rf*100,2)))
        rf_acc.append(acc_rf)
        
        classifier = RandomForestClassifier(n_estimators = 11,max_features = 6, criterion = 'entropy', random_state = 10,max_depth = 5)
        classifier.fit(x_train, y_train)
        op = classifier.predict(x_test)
        acc_sk = metrics.accuracy_score(y_test, op)
        print('\nAccuracy for random_forest (Sklearn)  = {}%'.format(round(acc_sk*100,2)))
        sk_rf_acc.append(acc_sk)
    
    print('\n------------------------------------------------------')
    print('\nMean Accuracy for random_forest (Implemented) after {} random splits  = {}%'.format(n,round(np.array(rf_acc).mean()*100,3)))
    print('\nMean Accuracy for random_forest (Sklearn) after {} random splits  = {}%'.format(n,round(np.array(sk_rf_acc).mean()*100,3)))
    
if __name__ == "__main__":

    print('\n-------------Random Forest (Implementation from scratch) for single random split -------------------')
    r = random_forest(data = train_df,total_trees = 11,tree_depth = 5,random_seed = 10,feature_per_tree = 6,bootstrap_size=100)
    
    r.fit()
    op= r.predict(x_test)
    prob_arr = r.predict_prob(x_test)
    pd.DataFrame({'Actual': y_test,'Predicted':op}).to_csv('Results\\Testing_Output.csv',index=False)
    
    print("\nCheck 'Results\\Testing_Output.csv' to compare actual and predicted values.")
    r.write_meta_data('Results\\Random_Forest_Meta_Data_Generated.xlsx')        
    
    print('\nConfusion Matrix \n')
    cm = confusion_matrix(y_test, op)
    print(pd.DataFrame(cm,columns=train_df.iloc[:,-1].unique(),index = train_df.iloc[:,-1].unique() ))
    
    
    acc = metrics.accuracy_score(y_test, op)
    print('\nAccuracy = {}%\n'.format(round(acc*100,2)))
    
    print('-------------Random Forest (Implementation from Sklearn) for single random split --------------------')
    
    
    classifier = RandomForestClassifier(n_estimators = 11,max_features = 6, criterion = 'entropy', random_state = 10,max_depth = 5)
    classifier.fit(x_train, y_train)
    op = classifier.predict(x_test)
    
    print('\nConfusion Matrix\n')
    cm_sk = confusion_matrix(y_test, op)
    print(pd.DataFrame(cm_sk,columns=train_df.iloc[:,-1].unique(),index = train_df.iloc[:,-1].unique() ))
    
    acc_sk = metrics.accuracy_score(y_test, op)
    print('\nAccuracy = {}%'.format(round(acc_sk*100,2)))
    
    #Uncomment last 2  lines to get mean accuracy after 10 random splits'
    print('\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Model Testing >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    gen_random_splits(10)
    
