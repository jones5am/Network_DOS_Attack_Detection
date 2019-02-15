# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 20:17:45 2018

@author: andrew.jones
"""

#%%Importing all Packages and Modules that I'll need
#Also setting the PATH varible for Graphviz to Work

import pandas as pd
import sklearn as sk
import numpy as np
from sklearn import tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os     

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'




#%%Reading the datasets in

Train = pd.read_csv('C:/Users/andrew.jones/Documents/Grad School/Machine Learning 1/Final Project Decided/NSL KDD2.csv', sep = ',', header = None)
Test = pd.read_csv('C:/Users/andrew.jones/Documents/Grad School/Machine Learning 1/Final Project Decided/NSL KDD Test.csv', sep = ',', header = None)


#%% Droping field that has no documenation around it's meaning

Train = Train.drop([42], axis = 1)
Test = Test.drop([42], axis = 1)

#%% Naming the columns

Train.columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins',
'logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack_type'] 


Test.columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins',
'logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack_type'] 

#%% Exploratory Data Analysis - different types of DOS Attacks

Attack_Types_Labels = ['neptune','smurf','back','land','pod','teardrop']

Attack_Types_Count = Train['attack_type'].value_counts()
Attack_Types_Count = Attack_Types_Count[[1,5,7,14,10,8]]
Attack_Types_Count.sort_values(ascending = False, inplace = True)

plt.bar(Attack_Types_Count.index.values, Attack_Types_Count, align = 'center', alpha = 0.5, width = 0.8, color = 'green')

plt.ylabel('Number of Records')
plt.title('Saturation by Type of DOS Attack')
plt.x_range = FactorRange(factors=df['label'].tolist())

plt.show()

#%% Encode the target variable as binary for just Denial of Service Attacks "Neptune" & "Smurf"

Train.loc[Train.attack_type == 'neptune','attack_type'] = 1
Train.loc[Train.attack_type == 'smurf','attack_type'] = 1
Train.loc[Train.attack_type == 'back','attack_type'] = 1
Train.loc[Train.attack_type == 'land','attack_type'] = 1
Train.loc[Train.attack_type == 'pod','attack_type'] = 1
Train.loc[Train.attack_type == 'teardrop','attack_type'] = 1
Train.loc[Train.attack_type == 'apache2','attack_type'] = 1
Train.loc[Train.attack_type == 'udpstorm','attack_type'] = 1
Train.loc[Train.attack_type == 'processtable','attack_type'] = 1
Train.loc[Train.attack_type == 'worm','attack_type'] = 1

Train.loc[Train.attack_type != 1, 'attack_type'] = 0 

Test.loc[Test.attack_type == 'neptune','attack_type'] = 1
Test.loc[Test.attack_type == 'smurf','attack_type'] = 1
Test.loc[Test.attack_type == 'back','attack_type'] = 1
Test.loc[Test.attack_type == 'land','attack_type'] = 1
Test.loc[Test.attack_type == 'pod','attack_type'] = 1
Test.loc[Test.attack_type == 'teardrop','attack_type'] = 1
Test.loc[Test.attack_type == 'apache2','attack_type'] = 1
Test.loc[Test.attack_type == 'udpstorm','attack_type'] = 1
Test.loc[Test.attack_type == 'processtable','attack_type'] = 1
Test.loc[Test.attack_type == 'worm','attack_type'] = 1

Test.loc[Test.attack_type != 1, 'attack_type'] = 0 
#%% Implement One Hot Encoding (OHE) on the Categorical Variables

Train_OHE= pd.get_dummies(Train[['protocol_type','flag']], drop_first = True)
Test_OHE = pd.get_dummies(Test[['protocol_type','flag']], drop_first = True)

#%% Concatenate the OHE variables back with the Original Features

Test_final = pd.concat([Test, Test_OHE], axis = 1)
Train_final = pd.concat([Train, Train_OHE], axis = 1)

#%% Select only the top 10 features that have been tested to be important (and the target for now)
Test_final = Test_final[['flag_S0','srv_serror_rate','dst_host_same_srv_rate','count','dst_host_srv_serror_rate','dst_bytes','src_bytes','diff_srv_rate','srv_count','dst_host_serror_rate','attack_type']]
Train_final = Train_final[['flag_S0','srv_serror_rate','dst_host_same_srv_rate','count','dst_host_srv_serror_rate','dst_bytes','src_bytes','diff_srv_rate','srv_count','dst_host_serror_rate','attack_type']]

#%%Exploratory Data Analysis - total DOS Attacks

Train_Length = len(Train_final)
DOS_Attacks_Count = sum(Train_final['attack_type'])
Saturation = [DOS_Attacks_Count, Train_Length]
Labels = ['Total Network Interactions', 'DOS Attacks']

plt.bar(Labels, Saturation, align = 'center', alpha = 0.5, width = 0.8, color = 'green')
plt.xticks(Labels)
plt.ylabel('Number of Records')
plt.title('Saturation of DOS Attacks')


plt.show()

#%%Using Principle Component Analysis to see if it gives superior performance 

### NOTE THIS DID NOT GIVE SUPERIOR PERFORMANCE AND IS THEREFORE COMMENTED OUT, BUT LEFT AS EVIDENCE OF TRYING ###

#stdsc = StandardScaler()
#Train_final_target_placeholder = Train_final.loc[:,'attack_type']
#Train_final = Train_final.drop('attack_type',axis = 1)
#Train_final = pd.DataFrame(stdsc.fit_transform(Train_final.iloc[:,:]))



#Test_final_target_placeholder = Test_final.loc[:, 'attack_type']
#Test_final = Test_final.drop('attack_type', axis = 1)
#Test_final = pd.DataFrame(stdsc.transform(Test_final.iloc[:,:]))


#pca = PCA(n_components = 8)

#Train_final = pd.DataFrame(pca.fit_transform(Train_final))
#Test_final = pd.DataFrame(pca.transform(Test_final))

#Train_final = pd.concat([Train_final, Train_final_target_placeholder], axis = 1)
#Test_final = pd.concat([Test_final, Test_final_target_placeholder], axis = 1)

#%%Splitting the Training & Testing Datasets into the independent and dependent dataframes / series

Train_Independent = Train_final.drop(labels = 'attack_type', axis = 1)
Train_Dependent = Train_final['attack_type']

Test_Independent = Test_final.drop(labels = 'attack_type', axis = 1)
Test_Dependent = Test_final['attack_type']

#%%Creating the decision trees based on both the entropy & gini coefficient criterions

decision_tree_entropy = tree.DecisionTreeClassifier(
        criterion = 'entropy',
        #max_features = 2,
        max_depth = 5,
        random_state = 1000        
        )

decision_tree_gini = tree.DecisionTreeClassifier(
        criterion = 'gini',
        #max_features = 2,
        max_depth = 5,
        random_state = 1000        
        )

#%%Fitting the models on the training data

decision_tree_entropy.fit(Train_Independent, Train_Dependent)
decision_tree_gini.fit(Train_Independent, Train_Dependent)


#%% Predicting our entropy models agains the Test Data and printing the confusion matrix

DOS_Prediction_E = decision_tree_entropy.predict(Test_Independent)
Confusion_Matrix_E = sk.metrics.confusion_matrix(Test_Dependent, DOS_Prediction_E)

print(Confusion_Matrix_E)

tn, fp, fn, tp = confusion_matrix(Test_Dependent, DOS_Prediction_E).ravel()
Accuracy_DT_E = (tn + tp)/len(Test_Dependent)

(tn, fp, fn, tp)

#%%Predicting our Gini coefficient model against the test data and printing the confusion matrix
DOS_Prediction_G = decision_tree_gini.predict(Test_Independent)
Confusion_Matrix_G = sk.metrics.confusion_matrix(Test_Dependent, DOS_Prediction_G)

tn, fp, fn, tp = confusion_matrix(Test_Dependent, DOS_Prediction_G).ravel()
Accuracy_DT_G = (tn + tp)/len(Test_Dependent)

(tn, fp, fn, tp)


#%%Exporting an image of the tree (chose entropy in this case, although both are obviously very similar)

dot_data = export_graphviz(decision_tree_entropy, filled = True, rounded = True,
                           class_names = ['DOS Attack', 'Not a DOS Attack'],
                           feature_names = Train_Independent.columns[0:],
                           out_file = None)


graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')


#%% Expanding the tree method into using an ensemble Random Forest Classifier using entropy in this case

forest_entropy = RandomForestClassifier(criterion = 'entropy', n_estimators = 10, random_state = 1, n_jobs =2)
forest_entropy.fit(Train_Independent, Train_Dependent)

#%% Predicting our entropy measured random forest model against the test data

DOS_Prediction_RF_E = forest_entropy.predict(Test_Independent)
Confusion_Matrix_RF_E = sk.metrics.confusion_matrix(Test_Dependent, DOS_Prediction_RF_E)

print(Confusion_Matrix_RF_E)

tn, fp, fn, tp = confusion_matrix(Test_Dependent, DOS_Prediction_RF_E).ravel()

Accuracy_RF_E = (tn + tp)/len(Test_Dependent)
(tn, fp, fn, tp)

#%% Relative Feature Importance Extraction
feature_importance = forest_entropy.feature_importances_
indices = np.argsort(feature_importance)[::-1]

print("Feature ranking:")

for f in range(Train_Independent.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], feature_importance[indices[f]]))

#%% Expanding the tree method using an ensemble Random Forest Classifier using the gini coefficient in this case

forest_gini = RandomForestClassifier(criterion = 'gini', n_estimators = 10, random_state = 1, n_jobs = 2, max_depth = 10)
forest_gini.fit(Train_Independent, Train_Dependent)

#%% Predicting our gini coefficient measured random forest model against the test data

DOS_Prediction_RF_G = forest_gini.predict(Test_Independent)
Confusion_Matrix_RF_G = sk.metrics.confusion_matrix(Test_Dependent, DOS_Prediction_RF_G)

print(Confusion_Matrix_RF_G)

tn, fp, fn, tp = confusion_matrix(Test_Dependent, DOS_Prediction_RF_G).ravel()

Accuracy_RF_G = (tn + tp)/len(Test_Dependent)
(tn, fp, fn, tp)


#%% Using Extra Tree Classifier with even more randomness in deciding the splits
    
forest_extrarandom = ExtraTreesClassifier(criterion = 'gini',n_estimators=10, max_depth=None, min_samples_split=2, random_state=1)

forest_extrarandom.fit(Train_Independent, Train_Dependent)

#%% Predicting our gini coefficient measured extra random forest model against the test data
DOS_Prediction_ERF = forest_extrarandom.predict(Test_Independent)
Confusion_Matrix_ERF = sk.metrics.confusion_matrix(Test_Dependent, DOS_Prediction_ERF)

print(Confusion_Matrix_ERF)

tn, fp, fn, tp = confusion_matrix(Test_Dependent, DOS_Prediction_ERF).ravel()

Accuracy_ERF = (tn + tp)/len(Test_Dependent)
(tn, fp, fn, tp)

#%% Using Gradient Boosting Classifier

forest_gradientboosting = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
...     max_depth=6, random_state=0).fit(Train_Independent, Train_Dependent)

#%% Predicting our gradient boosting random forest model against the test data
DOS_Prediction_GB = forest_gradientboosting.predict(Test_Independent)
Confusion_Matrix_GB = sk.metrics.confusion_matrix(Test_Dependent, DOS_Prediction_GB)

print(Confusion_Matrix_GB)

tn, fp, fn, tp = confusion_matrix(Test_Dependent, DOS_Prediction_GB).ravel()

Accuracy_GB = (tn + tp)/len(Test_Dependent)
(tn, fp, fn, tp)

#%% Print all Accuracy Scores

print(Accuracy_DT_E)
print(Accuracy_DT_G)
print(Accuracy_RF_E)
print(Accuracy_RF_G)
print(Accuracy_ERF)
print(Accuracy_GB)



