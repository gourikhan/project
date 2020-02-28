#!/usr/bin/env python
# coding: utf-8

# # Final Project - Bike Renting
# Gouri Khan
# January 26,2020

# # Load required libraries

# In[1]:


import os #To interact with local system directories
import pandas as pd # For data processing, CSV file import export (e.g. pd.read_csv)
import numpy as np  # for linear Algebra
from  matplotlib import pyplot
import matplotlib.pyplot as plt # For plotting and visualization
get_ipython().run_line_magic('matplotlib', 'inline')
# or inline plots in jupyter notebook


# In[2]:


import seaborn as sns # For plotting and visualization
sns.set(color_codes=True) # settings for seaborn plotting style
sns.set(rc={'figure.figsize':(6,6)}) # settings for seaborn plot size
from scipy.stats import uniform # import uniform distribution


# In[3]:


os.chdir("/Users/gourikhan/Desktop/Gouri") ##set the current working directory
os.getcwd() ##set the current working directory


# # Loading Dataset

# In[165]:


df_day = pd.read_csv("day.csv") ##load Bike rental data in PYTHON
df_day.head() #Print the top 5rows of the dataframe


# In[5]:


df_day.shape #understanding shape of data
#It contains (731 rows, 16 columns)


# In[6]:


df_day.info() #data  consist of all non-null Integers , Float and Object(categorical) variables.
#df_day.dtypes


# In[7]:


df_day.describe()


# In[8]:


# iterating the columns to get column names:
df_day.columns


# # exploratory data analysis

# In[166]:


import datetime
d1=df_day['dteday'].copy()
for i in range (0,d1.shape[0]):
    d1[i]=datetime.datetime.strptime(d1[i], '%Y-%m-%d').strftime('%d')
df_day['dteday']=d1


# In[167]:


#Feature Engineering
#Converting respective variables to required data format :

df_day['dteday']=df_day['dteday'].astype('category')
df_day['season']= df_day['season'].astype('category')
df_day['mnth']=df_day['mnth'].astype('category')

df_day['yr'] = df_day['yr'].astype('category')
df_day['holiday'] = df_day['holiday'].astype('category')
df_day['workingday'] = df_day['workingday'].astype('category')

df_day['weekday']=df_day['weekday'].astype('category')
df_day['weathersit']=df_day['weathersit'].astype('category')

df_day = df_day.drop(['instant'], axis=1)
#we can drop instant column as this is only index values.


# # Missing value analysis

# In[12]:


total_missing_value = df_day.isnull().sum()
total_missing_value

#CONCLUSION : There is no missing value in the dataframe.
#Also as per df_day.info() result all variavles are non-null so we can conclude there is no missing value in dataset.


# # Outlier Analysis

# In[13]:


plt.boxplot(df_day['temp']) #box plot of temp variable


# In[14]:


plt.boxplot(df_day['atemp']) #box plot of atemp variable


# In[15]:


plt.boxplot(df_day['hum']) #box plot of humidity variable


# In[16]:


plt.boxplot(df_day['windspeed']) #box plot of windspeed variable


# In[17]:


plt.boxplot(df_day['casual']) #box plot of casual variable


# In[18]:


plt.boxplot(df_day['registered']) #box plot of registered variable


# we can see there is very less outliers in hum and few in windspeed but this are weather variables and we thing this might be due to seasonal condition so will not remove any outliers.
# for casual there is a huge outliers that may be because of there is no normality in the variable
# so first we need to check normality for numerical variable the will decide removing outliers.

# # Univariant analysis for numerical variables

# # feature  Scaling : Normality  Check

# In[19]:


# Target variable  analysis
sns.distplot(df_day['cnt']) #Check whether target variable is normal or not
print("Skewness of target variable: %f" % df_day['cnt'].skew())
print("Kurtosis of target variable: %f" % df_day['cnt'].kurt())
#Skewness is very low,so target variable is normally distributed
df_day['cnt'].describe() #descriptive statistics summary


# In[20]:


#Distribution  independent numeric variables 
sns.distplot(df_day['temp']) #Check whether  variable 'temp' is normal or not
print("Skewness of temp: %f" % df_day['temp'].skew())
print("Kurtosis of temp: %f" % df_day['temp'].kurt())
df_day['temp'].describe()


# In[21]:


sns.distplot(df_day['atemp']) #Check whether  variable 'atemp' is normal or not
print("Skewness of atemp: %f" % df_day['atemp'].skew())
print("Kurtosis of atemp: %f" % df_day['atemp'].kurt())
df_day['atemp'].describe()


# In[22]:


sns.distplot(df_day['hum']) #Check whether  variable 'hum' is normal or not
print("Skewness of humidity: %f" % df_day['hum'].skew())
print("Kurtosis of humidity: %f" % df_day['hum'].kurt())
df_day['hum'].describe()


# In[23]:


sns.distplot(df_day['windspeed']) #Check whether  variable 'windspeed'is normal or not
print("Skewness of windspeed: %f" % df_day['windspeed'].skew())
print("Kurtosis of windspeed: %f" % df_day['windspeed'].kurt())
df_day['windspeed'].describe()


# In[24]:


sns.distplot(df_day['casual']) #Check whether  variable 'casual'is normal or not
print("Skewness of casual: %f" % df_day['casual'].skew())
print("Kurtosis of casual: %f" % df_day['casual'].kurt())
df_day['casual'].describe()


# In[25]:


sns.distplot(df_day['registered']) #Check whether  variable 'casual'is normal or not
print("Skewness of registered: %f" % df_day['registered'].skew())
print("Kurtosis of registered: %f" % df_day['registered'].kurt())
df_day['registered'].describe()


# In[168]:


# feature  Scaling NORMALIZATION
cnames = ['casual','registered']
for i in cnames :
    df_day[i] = (df_day[i] - min(df_day[i]))/(max(df_day[i]) - min(df_day[i]))


# # Bivariant analysis for numerical variables

# In[27]:


#box plot of 'Weekdays' with target 'cnt'
data = pd.concat([df_day['cnt'], df_day["weekday"]], axis=1)
f, ax = plt.subplots(figsize=(10, 6)) #Set the width and hieght of the plot
fig = sns.boxplot(x="weekday", y="cnt", data=data)
fig.axis(ymin=0, ymax=9000);

#CONCLUSION: below Boxplot is saying that for all the weekdays median in between 4000- 5000


# In[28]:


#box plot of 'holiday' with target 'cnt'
data = pd.concat([df_day['cnt'], df_day['holiday']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='holiday', y="cnt", data=data)
fig.axis(ymin=0, ymax=9000);
df_day['holiday'].value_counts()

#CONCLUSION: below Boxplot is saying that median high on non-holidays but on holidays range is huge


# In[29]:


#box plot of'workingday' with target 'cnt'
data = pd.concat([df_day['cnt'], df_day['workingday']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='workingday', y="cnt", data=data)
fig.axis(ymin=0, ymax=9000);
df_day['workingday'].value_counts()

#CONCLUSION: below Boxplot is saying that median is same almost when compare to workingday


# In[30]:


#box plot of 'weathersit' with target 'cnt'
data = pd.concat([df_day['cnt'], df_day['weathersit']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='weathersit', y="cnt", data=data)
fig.axis(ymin=0, ymax=9000);

#CONCLUSION: below Boxplot is saying that as the weather is good moderate and bad bike renting is varies accordingly


# In[31]:


#box plot of 'season' with target 'cnt'
data = pd.concat([df_day['cnt'], df_day['season']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='season', y="cnt", data=data)
fig.axis(ymin=0, ymax=9000);

#CONCLUSION: below Boxplot is saying that median is hign on summer and fall season and lowest on spring


# In[32]:


#box plot of 'year' with target 'cnt'
data = pd.concat([df_day['cnt'], df_day['yr']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='yr', y="cnt", data=data)
fig.axis(ymin=0, ymax=9000);
df_day['yr'].value_counts()

#CONCLUSION: below Boxplot is saying that median  higher in 2012, but total count is same for both years


# In[33]:


#box plot of 'month' with target 'cnt'
data = pd.concat([df_day['cnt'], df_day['mnth']], axis=1)
f, ax = plt.subplots(figsize=(15, 6))
fig = sns.boxplot(x='mnth', y="cnt", data=data)
fig.axis(ymin=0, ymax=9000);

#CONCLUSION: below Boxplot is saying that median is varying in every month but not the counts


# In[34]:


#box plot of 'dteday' with target 'cnt'
data = pd.concat([df_day['cnt'], df_day['dteday']], axis=1)
f, ax = plt.subplots(figsize=(20, 6))
fig = sns.boxplot(x='dteday', y="cnt", data=data)
fig.axis(ymin=0, ymax=9000);


# # Feature  selection

# In[35]:


#Feature selection on the basis of various features like correlation, multicollinearity.
cnames = df_day.columns
df_corr = df_day.loc[:,cnames]
f, ax = plt.subplots(figsize=(9, 6))
#Generate correlation matrix
corr = df_corr.corr()
#Plot heatmap using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(100, 10, as_cmap=True),
            square=True, ax=ax)

#correlation matrix among all numeric variables and analyse what are the important variables using pearson corelation 
df_corr.corr(method='pearson').style.format("{:.3}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# In[36]:


# check relationship of all numeric variables with each other with pair plots
numerical_columns = ["temp","atemp","hum","windspeed","casual","registered","cnt"]
sns.set()
sns.pairplot(df_corr[numerical_columns], height = 5,kind="reg")
plt.show();


# CONCLUSION :As per above scatter plots and Correlation graph there is strong relation between Independent variable 
# 'temp' and 'atemp' so we can discard one of them as they are carrying same information or can take average to a new
# column and discard both.
# variable hum is very less correlated with target cnt, so we can discard this too.

# # Chi Square Test of Independence for Categorical variables

# In[37]:


import scipy
from scipy import stats #import chi2_contigency #  for Chi square Test
from scipy.stats import chi2
from scipy.stats import chi2_contingency 


# In[38]:


cat_names = df_day.columns
len(cat_names)
index=0;
no_of_col = len(cat_names)
relationship_matrix = [[0] * no_of_col for column in range(no_of_col)]
jIndex=0;
for j in cat_names:
    iIndex=0;
    strng=''
    for i in cat_names:
        alpha = 0.05 #Significance Level 5%
        chi2, p, dof ,ex = chi2_contingency(pd.crosstab(df_day[j], df_day[i]))
        critical_value=stats.chi2.ppf(q=1-alpha,df=dof)
        if chi2>=critical_value and p<=alpha:
            #string 1 means there is relation between variables
            strng = "1"
        else:
            #string 0 means there is no relation between variables
            strng = "0"
        Q = [chi2, critical_value, p, alpha , dof]
        relationship_matrix[iIndex][jIndex] = strng
        iIndex=iIndex+1
    jIndex=jIndex+1  


# In[39]:


data = pd.DataFrame(relationship_matrix)
data = data.rename(columns={0:"dteday",1:"season",2:"yr",3:"mnth",4:"holiday",5:"weekday",6:"workingday",7:"weathersit",8:"temp",9:"atemp",10:"hum",11:"windspeed",12:"casual",13:"registered",14:"cnt"})
data = data.rename(index={0:"dteday",1:"season",2:"yr",3:"mnth",4:"holiday",5:"weekday",6:"workingday",7:"weathersit",8:"temp",9:"atemp",10:"hum",11:"windspeed",12:"casual",13:"registered",14:"cnt"})
data


# There is a relationship between variable : season mnth
# There is a relationship between variable : season weathersit
# There is a relationship between variable : season temp
# There is a relationship between variable : season casual
# There is a relationship between variable : mnth weathersit
# There is a relationship between variable : mnth temp
# There is a relationship between variable : temp weathersit
# There is a relationship between variable : hum weathersit
# There is a relationship between variable : atemp temp
# There is a relationship between variable : hum temp
# There is a relationship between variable : hum windspeed
# There is a relationship between variable : windspeed temp
# There is a relationship between variable : holiday weekday
# There is a relationship between variable : holiday workingday
# There is a relationship between variable : weekday workingday
# so we can
# Remove weekday,holiday because this is co-related with workingday, removing season because this is correlated with weathersit and mnth,
# dteday variable as this variable has no impact on output.

# In[169]:


#Create a new column of the mean of temp and atemp.
df_day["mean_temp_atemp"] = (df_day["temp"]+ df_day["atemp"])/2


# In[198]:


df_day = df_day.drop(['dteday','temp','atemp','weekday','hum','season','holiday'], axis =1)


# # Spliting Test and train data using skilearn train_test_split 

# In[171]:


X = df_day.loc[:, df_day.columns != 'cnt']
Y = df_day['cnt']


# In[172]:


from sklearn.model_selection import train_test_split ,cross_val_score
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=25)


# # Decision Tree Regressor 

# In[173]:


#Importing Decision Tree Regressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


# In[174]:


#Calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape

#Calculate RMSE
def RMSE(y_test,y_predict):
    mse = np.mean((y_test-y_predict)**2)
    rmse=np.sqrt(mse)
    return rmse


# In[175]:


max_depth = 9
min_samples_split =4
tree = DecisionTreeRegressor(max_depth =max_depth , min_samples_split =min_samples_split, random_state = 1)
DT_model = tree.fit(X_train, Y_train)
print(DT_model)


# In[176]:


predictions_DT = DT_model.predict(X_test)


# In[177]:


MAPE(Y_test,predictions_DT)


# In[178]:


RMSE(Y_test,predictions_DT)


# In[179]:


metrics.mean_absolute_error(Y_test, predictions_DT)


# In[180]:


metrics.mean_squared_error(Y_test, predictions_DT)


# # Decision tree plot using pydotplus and graphviz:

# In[200]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import graphviz


# In[207]:


dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[211]:


#graph


# # Random Forest

# In[181]:


# Import random forest regressor from sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor


# In[277]:


#Random forest model building
RF_model = RandomForestRegressor(n_estimators= 130, random_state=100).fit(X_train,Y_train)
print(RF_model)


# In[278]:


# Predict the model using predict funtion
RF_predict= RF_model.predict(X_test)


# In[279]:


#Evaluate Random forest using  MAPE 
MAPE(Y_test,RF_predict)


# In[280]:


#Evaluate  Model usinf  RMSE
RMSE(Y_test,RF_predict)


# In[281]:


#Mean Absolute Error (MAE)
metrics.mean_absolute_error(Y_test, RF_predict)


# In[282]:


#Mean Squared Error (MSE) 
metrics.mean_squared_error(Y_test, RF_predict)


# # Linear Regression

# In[188]:


#import required library for linear regreesion  
import statsmodels.api as sm


# In[189]:


cnames = df_day.columns
df_day[cnames] = df_day[cnames].apply(pd.to_numeric, errors='coerce', axis=1)


# In[190]:


X = df_day.loc[:, df_day.columns != 'cnt']
Y = df_day['cnt']


# In[191]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=25)


# In[192]:


#develop Linear Regression model using sm.ols
LR_model = sm.OLS(Y_train,X_train).fit()
LR_model


# In[193]:


#predict target using LR model
predict_LR = LR_model.predict(X_test)


# In[194]:


# Print out the statistics
LR_model.summary()


# In[195]:


#Predict the model using  RMSE
print(RMSE(Y_test,predict_LR))


# In[196]:


#evaluate model using MAPE
print(MAPE(Y_test,predict_LR))


# it is  showing that  Linear Regression model is  best suitable for the dataset.
# Conclusion : Linear regression is the  best model for the dataset.
