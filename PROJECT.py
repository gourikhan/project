#import all required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #for getting working directory 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.model_selection import GridSearchCV , train_test_split ,cross_val_score 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , f1_score , make_scorer , mean_absolute_error ,confusion_matrix ,roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler,OneHotEncoder , LabelEncoder ,normalize
from sklearn.naive_bayes import GaussianNB #Naive Bayes classifier algorithms
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier
import scipy
from scipy import stats


#setting directory
#os.chdir("C:\\Users\\Gouri\\Desktop\\gouri\\data sciece study") #os.chdir means library.function of that library
#check current working directory
os.getcwd()
#1. Reading Datasets
#load .csv data in python
train=pd.read_csv("train.csv",sep=',')
test=pd.read_csv("test.csv",sep=',')
#2. Data Analysis
train.drop(['ID_code'] , inplace=True , axis=1)
test.drop(['ID_code'] , inplace=True , axis=1)
# mark zero values as missing or NaN
train.iloc[:,1:201] = train.iloc[:,1:201].replace(0, np.NaN)
# count the number of NaN values in each column
#print(train.isnull().sum())
#Remove Rows With Missing Values because we have a huge data seta nd missing values are very less
train.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print(train.shape) 
#3. Data Preprocessing
for col in train.columns: 
    #print(col) 
    #print((train[[col]] == 0).sum())
    sns.boxplot(x=train[[col]])	
#Z-Score-
#The Z-score is the signed number of standard deviations by which the value of an observation or data point is above the mean 
#value of what is being observed or measured.
z = np.abs(scipy.stats.zscore(train))
print(z)
threshold = 3
print(np.where(z > 3))
train_o = train[(z < 3).all(axis=1)]
train.shape,train_o.shape
#Feature & Target Set
X = train_o.iloc[:,1:201].values  #this is my x_train
Y = train_o.iloc[:, 0].values #this is my y_train #Length: 188835,
#Feature & Target Set
#test["TARGET"] = ""   #New column in test data named as TARGET
# mark zero values as missing or NaN
#test.iloc[:,-1] = test.iloc[:,-1].replace("", np.NaN)
x = test.iloc[:,:200].values  #this is my x_train
#y = test.iloc[:,-1].values #this is my y_train
#training and testing split for train dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=25)
#logostic regression
model_lgr = LogisticRegression(max_iter = 200)
model_lgr.fit(X_train,Y_train)
print("model_lgr accuracy is:{}".format(model_lgr.score(X_test,Y_test)))  #our accuracy is:0.9148823498261284
roc=roc_auc_score(Y_test, lgr.predict_proba(X_test)[:,1])
print("AUC score for lgr is:{}".format(roc))
#We can evaluate our model so and we have y_predict and y_true(y_test)
Y_true = Y_test 
Y_pred_lgr = model_lgr.predict(X_test) #Predict data for eveluating 
cm_lgr = confusion_matrix(Y_true,Y_pred_lgr)
#cm_lgr
average_precision = average_precision_score(Y_test, Y_pred)
print('Average precision-recall score for lgr model: {0:0.2f}'.format(average_precision))
#from sklearn.metrics import precision_recall_fscore_support
F1score_macro_lgr = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
print("F1score_macro_lgr is:{}".format(F1score_macro_lgr))
F1score_micro_lgr = precision_recall_fscore_support(Y_test, Y_pred, average='micro')
print("F1score_micro_lgr is:{}".format(F1score_micro_lgr))
F1score_weighted_lgr = precision_recall_fscore_support(Y_test, Y_pred, average='weighted')
print("F1score_weighted_lgr is:{}".format(F1score_weighted_lgr))
MAE_lgr = mean_absolute_error(Y_test, Y_pred)
print("mean_absolute_error for model_lgr is:{}".format(MAE_lgr))
#We draw heatmap for showing confusion matrix
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(cm_lgr,annot = True,linewidth = 2,fmt =".0f",ax = ax)
test_pred_Final = model_lgr.predict(x)
test_pred_Final.shape
test_pred_Final
test['TARGET_lgr'] = pd.DataFrame(test_pred_Final)  
new_series = pd.Series(test_pred_Final)
test['TARGET_lgr'] .value_counts()

#Naive Bayes classifier algorithms:::
#Rather than attempting to calculate the probabilities of each attribute value, they are assumed to be conditionally independent
#given the class value.
#This is a very strong assumption that is most unlikely in real data, i.e. that the attributes do not interact. 
#Nevertheless, the approach performs surprisingly well on data where this assumption does not hold.
#Create a Gaussian Classifier
model_gn = GaussianNB()
# Train the model using the training sets 
model_gn.fit(X_train, Y_train)
#Predict Output 
#predicted= model.predict([[Y_test]])
#print (predicted)
print("model_gn accuracy is:{}".format(model_gn.score(X_test,Y_test)))
#We can evaluate our model so and we have y_predict and y_true(y_test)
Y_pred = model_gn.predict(X_test) #Predict data for eveluating 
cm_gn = confusion_matrix(Y_true,Y_pred)
#We draw heatmap for showing confusion matrix
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(cm_gn,annot = True,linewidth = 2,fmt =".0f",ax = ax)
#cm_gn
roc=roc_auc_score(Y_test, model_gn.predict_proba(X_test)[:,1])
print("AUC score for model_gn is:{0:0.2f}".format(roc))
average_precision = average_precision_score(Y_test, Y_pred)
print('Average precision-recall score for model_gn: {0:0.2f}'.format(average_precision))
#from sklearn.metrics import precision_recall_fscore_support
F1score_macro_gn = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
print("F1score_macro_gn is:{}".format(F1score_macro_lgr))
F1score_micro_gn = precision_recall_fscore_support(Y_test, Y_pred, average='micro')
print("F1score_micro_gn is:{}".format(F1score_micro_lgr))
F1score_weighted_gn = precision_recall_fscore_support(Y_test, Y_pred, average='weighted')
print("F1score_weighted_gn is:{}".format(F1score_weighted_lgr))
MAE_gn = mean_absolute_error(Y_test, Y_pred)
print("mean_absolute_error for model_gn is:{}".format(MAE_gn))
test_pred_Final = model_gn.predict(x)
test['TARGET_gn'] = pd.DataFrame(test_pred_Final) 
test['TARGET_gn'].value_counts()

#Random Forest - Model 
model_rfc = RandomForestClassifier(random_state=42)
model_rfc.fit(X_train, Y_train)
print("model_rfc accuracy is:{}".format(model_rfc.score(X_test,Y_test)))
Y_true = Y_test 
Y_pred = model_rfc.predict(X_test)
cm_rfc = confusion_matrix(Y_true,Y_pred)
cm_rfc
rfc_score = model_rfc.score(X_test, Y_test)
print("model_rfc accuracy score is:{}".format(rfc_score))
#We draw heatmap for showing confusion matrix
#import matplotlib.pyplot as plt
#import seaborn as sns
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(cm_rfc,annot = True,linewidth = 2,fmt =".0f",ax = ax)
test_pred_Final = model_rfc.predict(x)
test_pred_Final
test['TARGET_rfc'] = pd.DataFrame(test_pred_Final)
test['TARGET_rfc'].value_counts()
MAE_rfc = mean_absolute_error(Y_test, Y_pred)
print("mean_absolute_error for model_rfc is:{}".format(MAE))
roc=roc_auc_score(Y_test, model_rfc.predict_proba(X_test)[:,1])
print("AUC score for model_rfc is:{0:0.2f}".format(roc))
average_precision = average_precision_score(Y_test, Y_pred)
print('Average precision-recall score for model_rfc: {0:0.2f}'.format(average_precision))
#from sklearn.metrics import precision_recall_fscore_support
F1score_macro_rfc = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
print("F1score_macro_rfc is:{}".format(F1score_macro_rfc))
F1score_micro_rfc = precision_recall_fscore_support(Y_test, Y_pred, average='micro')
print("F1score_micro_rfc is:{}".format(F1score_micro_rfc))
F1score_weighted_rfc = precision_recall_fscore_support(Y_test, Y_pred, average='weighted')
print("F1score_weighted_rfc is:{}".format(F1score_weighted_rfc))

#FINAL PREDICTED TEST DATASET SAVED IN FILE PATH:
test.to_csv(r'C:\\Users\\Gouri\\Desktop\\gouri\\data sciece study\\final_test.csv', index=False) 
