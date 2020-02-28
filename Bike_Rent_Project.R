#*************************Load required libraries and packages****************************

rm(list=ls()) #remove everythng from R, to clear RAM

setwd("/Users/gourikhan/Desktop/Gouri") #set the current working directory
getwd() #get current working directory

#Install required packages 
x = c("ggplot2","corrgram","caret","randomForest","C50","e1071","rpart","sampling","GoodmanKruskal","usdm")
lapply(x, require, character.only = TRUE)
rm(x)

# Install  Require libraries
library(GoodmanKruskal)
library(corrgram)
library(usdm)
library(rpart)
library(rpart.plot)
#*****************************Loading Dataset************************************
#load Bike rental data in R
df_day= read.csv("day.csv", header = T)

#*****************************Exploratory Data Analysis****************************

summary(df_day) # Summarizing  data 
str(df_day) #structure of data
#target variable is 'cnt',rest of the variables are independent variable(or predictors) 

#It  shows variables like 'mnth',holiday','weekday','weathersit','season' are catogical variables and we need to change to correct variable types encoding
#Nummeric  vaiables like 'temp','atem','hum','windspeed' are given in normalized form

df_day$season=as.factor(df_day$season)
df_day$mnth=as.factor(df_day$mnth)
df_day$yr=as.factor(df_day$yr)
df_day$holiday=as.factor(df_day$holiday)
df_day$weekday=as.factor(df_day$weekday)
df_day$workingday=as.factor(df_day$workingday)
df_day$weathersit=as.factor(df_day$weathersit)

df_day=subset(df_day,select = -c(instant))

d1=unique(df_day$dteday)
df=data.frame(d1)
df_day$dteday=format(as.Date(df$d1,format="%Y-%m-%d"), "%d")
df_day$dteday=as.factor(df_day$dteday)

#***************************Missing Values Analysis***********************************
missing_val = data.frame(apply(df_day,2,function(x){sum(is.na(x))}))
missing_val #CONCLUSION : no missing  values are presnt in the data set

#******************************Outlier Analysis*****************************************
# BoxPlots - Distribution and Outlier Check in  numerical variables
numeric_index = sapply(df_day,is.numeric) #selecting only numeric
numeric_data = df_day[,numeric_index]
cnames = colnames(numeric_data)
#cnames

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt", group=1), data = subset(df_day))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "blue" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="cnt")+
           ggtitle(paste("Box plot of count for",cnames[i])))
}
# detect outliers using box plot
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,ncol=2)

#we can see there is very less outliers in hum and few in windspeed but this are weather variables and we thing this might be due to seasonal so will not remove any outliers.
#for casual there is a huge outliers that may be because of there is no normality in the variable
#so first we need to check normality for numerical variable the will decide removing outliers:

#*****************************univariate data analysis**********************************
# function to create univariate distribution of numeric  variables
univariate_numeric <- function(num_x) {
  ggplot(df_day)+
    geom_histogram(aes(x=num_x,y=..density..),
                   fill= "grey")+
    geom_density(aes(x=num_x,y=..density..))}

# analyze the distribution of  target variable 'cnt'
univariate_numeric(df_day$cnt)
# the above graph is showing 'cnt'  is normally distributed

# analyse the distrubution of  independence variable 'temp'
univariate_numeric(df_day$temp)

# analyse the distrubution of  independence variable 'atemp'
univariate_numeric(df_day$atemp)

# analyse the distrubution of  independence variable 'hum'
univariate_numeric(df_day$hum)

# analyse the distrubution of  independence variable 'windspeed'
univariate_numeric(df_day$windspeed)

# analyse the distrubution of  independence variable 'casual'
univariate_numeric(df_day$casual) 
## the above graph is showing 'casual'  is not at all normally distributed ,so we need to normalize this variable first

# analyse the distrubution of  independence variable 'registered' #normal distribution
univariate_numeric(df_day$registered)

#*************************feature  Scaling NORMALIZATION********************************
pnames = c("casual","registered")
for(i in pnames){
  df_day[,i] = (df_day[,i] - min(df_day[,i]))/
    (max(df_day[,i] - min(df_day[,i])))}

#****************************Bivariate data analysis***********************************
# Visualize categorical Variable 'mnth' with target variable 'cnt'
ggplot(df_day, aes(x=as.factor(mnth), y=cnt),fill="grey") + 
  stat_summary(fun.y="mean", geom="bar")

# Visualize categorical Variable 'yr' with target variable 'cnt'
ggplot(df_day, aes(x=as.factor(yr), y=cnt),fill="grey") + 
  stat_summary(fun.y="mean", geom="bar")

# Visualize categorical Variable 'dteday' with target variable 'cnt'
ggplot(df_day, aes(x=as.factor(dteday), y=cnt),fill="grey") + 
  stat_summary(fun.y="mean", geom="bar")

# Visualize categorical Variable 'season' with target variable 'cnt'
ggplot(df_day, aes(x=as.factor(season), y=cnt),fill="grey") + 
  stat_summary(fun.y="mean", geom="bar")

# Visualize categorical Variable 'holiday' with target variable 'cnt'
ggplot(df_day, aes(x=as.factor(holiday), y=cnt),fill="grey") + 
  stat_summary(fun.y="mean", geom="bar")

# Visualize categorical Variable 'weekday' with target variable 'cnt'
ggplot(df_day, aes(x=as.factor(weekday), y=cnt),fill="grey") + 
  stat_summary(fun.y="mean", geom="bar")

# Visualize categorical Variable 'workingday' with target variable 'cnt'
ggplot(df_day, aes(x=as.factor(workingday), y=cnt),fill="grey") + 
  stat_summary(fun.y="mean", geom="bar")

# Visualize categorical Variable 'weathersit' with target variable 'cnt'
ggplot(df_day, aes(x=as.factor(weathersit), y=cnt),fill="grey") + 
  stat_summary(fun.y="mean", geom="bar")

# Visualize categorical Variable 'holiday' 
ggplot(df_day) +
  geom_bar(aes(x=holiday),fill="grey")
# it is showing that almost all the  cycle rentals are happening  on holidays

# Visualize categorical Variable 'weekday' 
ggplot(df_day) +
  geom_bar(aes(x=weekday),fill="grey") 
# it is showing  counts are same on all weekdays

# Visualize categorical Variable 'workingday' 
ggplot(df_day) +
  geom_bar(aes(x=workingday),fill="grey") 
# it is showing  counts are very high on a workingday

# Visualize categorical Variable 'season' 
ggplot(df_day) +
  geom_bar(aes(x=season),fill="grey") 
# it is showing  counts are not varying as per season

# Visualize categorical Variable 'weathersit' 
ggplot(df_day) +
  geom_bar(aes(x=weathersit),fill="grey") 
# count aries according to whether is Clear, Few clouds, Partly cloudy, Partly cloudy.

# Visualize categorical Variable 'yr' 
ggplot(df_day) +
  geom_bar(aes(x=yr),fill="grey") 
# it is showing  counts are not varying as per yr

# Visualize categorical Variable 'dteday' 
ggplot(df_day) +
  geom_bar(aes(x=dteday),fill="grey") 
# it is showing  counts are not varying daily but the last few days of the month.

# Visualize categorical Variable 'mnth' 
ggplot(df_day) +
  geom_bar(aes(x=mnth),fill="grey") 
# it is showing  counts are not varying monthly

#*****************bivariate relationship between numeric variables**********************
#check the relationship between all numeric variable using pair plot
ggpairs(df_day[,cnames])

# verify correleation between Numeric variables
corrgram(df_day[,cnames], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

# that above plot stating that less relationship between cnt-hum.
# and there is strong positive relationship between temp-cnt and  atemp-cnt, 
#but temp and atemp are highly corelated.so we need to delete one variable to avoide multicolinearity or we can take mean value as a new variable and drop both the variables.

#****************visualize the relationship between categorical variable******************

cat_values<- c("holiday", "mnth", "season","yr","weekday","workingday","weathersit","cnt")
subset_df_day<- subset(df_day, select = cat_values)
GKmatrix<- GKtauDataframe(subset_df_day)
plot(GKmatrix, corrColors = "blue")

#********************Feature Selection or Dimension Reduction****************************
colnames(df_day)
#feature engineering to make new variavle out of mean of temp and atemp:
df_day$mean_temp_atemp = (df_day$temp + df_day$atemp)/2

df_day = subset(df_day,select=-c(dteday,temp,atemp,weekday,holiday,hum,season))

#************************Dividing data into Test Train variables************************
#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(df_day$cnt, p = .75, list = FALSE)
train = df_day[ train.index,]
test  = df_day[-train.index,]

target = subset(test, select= -c(cnt))

#*****************************Model Development******************************

#**************************develop Decision tree model******************************
#rpart for regression
DT = rpart(cnt ~ ., data = train, method = "anova")
#Predict for new test cases
predictions_DT = predict(DT, target)
print(DT)

#  plotting decision tree
rpart.plot(DT)
#*****************************Evaluation Matrix*************************************
#calculate MAPE
MAPE = function(y_true, y_pred){100 *
  mean(abs((y_true - y_pred)/y_true))}

#Evaluate  Model using RMSE
RMSE <- function(y_test,y_predict) {
  difference = y_test - y_predict
  root_mean_square = sqrt(mean(difference^2))
  return(root_mean_square)}

#*****************************Evaluate  Decision tree*********************************
MAPE(test$cnt, predictions_DT) 

RMSE(test$cnt, predictions_DT) 
#**************************Random Forest regressor*************************************
RF=randomForest(cnt ~ . , data = train)
RF
plot(RF)
predictions_RF = predict(RF, target)
#*************************Evaluate Random Forest algorithm*****************************
MAPE(test$cnt, predictions_RF) 

RMSE(test$cnt, predictions_RF) 
#***************************Parameter Tuning for random forest**************************

RF2=randomForest(cnt ~ . , data = train,mtry =7,ntree=130 ,nodesize =10 ,importance =TRUE)
RF2
predictions_RF2 = predict(RF2, target) #Predict for new test cases

MAPE(test$cnt, predictions_RF2) 
RMSE(test$cnt, predictions_RF2) 
#***************************check Variable  Importance*********************************
varimp <- importance(RF2)
varimp
# sort variables as per importance
sort_var <- names(sort(varimp[,1],decreasing =T))
sort_var
varImpPlot(RF2,type = 2) # draw varimp plot 

#*************************Develop  Linear Regression Model*******************************
LR_model = lm(cnt ~., data = train) #run regression model
summary(LR_model) #Summary of the model

predictions_LR = predict(LR_model, target) # Predict  the Test data 

#*************************Evaluate Linear Regression Model******************************
MAPE(test$cnt, predictions_LR) 

RMSE(test$cnt, predictions_LR)

# Conclusion  For this Dataset  Linear Regression is  Accuracy  is '99.9'
