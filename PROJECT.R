#*****************************************************************************************************************
#1 Settings and data import
rm(list=ls()) #remove everythng from R, to clear RAM
getwd() #get current working directory
#set current working directory as per your system path:
#setwd("C:/Users/Gouri/Desktop/desktop/PROJECT_folder/")

#install multiple package
#install.packages("pacman")
#install.packages("iterators","forecast","caret","ISLR","tree","randomForest","dplyr","Deducer","scorer","MLmetrics")
library(pacman)
gc()
pacman::p_load(pROC,Deducer,stats,base,dplyr,scorer,MLmetrics,forecast, ggplot2,e1071,class,readr,reader,rpart,modelr,randomForest,lattice,caret,ISLR,tree)

#2 load train and test .csv data in R from file storage
train = read.csv("train.csv",header=T)
test = read.csv("test.csv",header=T)
train$ID_code <- NULL
test$ID_code <- NULL
input <- train %>% select(2:201)
output <- train %>% select(1)

#3.In this challenge, we need to identify which customers will make a specific transaction in the future,irrespective of the amount of money transacted.

#4 Exploratory data analysis and data manipulation
input[input==0] <- NA
# Add datasets horizontally
totalDataset <- cbind(output, input)
NA_omit_dataset <- na.omit(totalDataset)
#5 Final data selection
n = nrow(NA_omit_dataset)
Split = sample(c(TRUE, FALSE), n, replace=TRUE, prob=c(0.7, 0.3))
training = NA_omit_dataset[Split, ]
testing = NA_omit_dataset[!Split, ]

training$target <- factor(training$target)
testing$target <- factor(testing$target)
rm(input,output,totalDataset,train)
#str(NA_omit_dataset)

#MODEL1: LOGISTIC REGRESSION:
#******************************************************************************************************
model_logicalRegression <- glm(target ~ ., data=training, family=binomial(link = 'logit'), maxit = 100)
# impliment prediction algorithm on the test set
predicted_prob <- predict(model_logicalRegression, testing, type='response')
predicted_outcome <- as.factor(ifelse(predicted_prob > .5, '1', '0'))
#predicted_outcome
# create confusion matrix
cf_lr <- confusionMatrix(data=predicted_outcome, reference=testing$target )
#cf_lr
summary(model_logicalRegression)
Accuracy_lr<-round(cf_lr$overall[1],2)
Accuracy_lr
precision_lr = precision(table(predicted_outcome,testing$target))
precision_lr
recall_lr = recall(table(predicted_outcome,testing$target))
recall_lr
rocplot(model_logicalRegression)
F1_Score(testing$target, predicted_outcome)
mean_absolute_error(testing$target, predicted_outcome)
test_pred_lr <- predict(model_logicalRegression, test, type='response')
test['TARGETlr'] = as.data.frame(test_pred_lr)

#MODEL:NAIVE BYES:
#**************************************************************************************************
nb_model <- naiveBayes(target ~ ., data = training, type="class")
nb_model$apriori
predictions_nb <- predict(nb_model, testing)
cf_nb <- confusionMatrix(data=predictions_nb, reference=testing$target )
#cf_nb
summary(nb_model)
Accuracy_nb<-round(cf_nb$overall[1],2)
Accuracy_nb
precision_nb = precision(table(predictions_nb,testing$target))
precision_nb
recall_nb = recall(table(predictions_nb,testing$target))
recall_nb
F1_Score(testing$target, predictions_nb)
mean_absolute_error(testing$target, predictions_nb)
test_pred_nb <- predict(nb_model, test)
test['TARGETnb'] = as.data.frame(test_pred_nb)

#MODEL: RANDOM FOREST:
#***************************************************************************************************
#install the caret package and the randomForest package.
#Next step is to load the packages into the working environment.
set.seed(1234)
model_rf <- randomForest(target ~., data = training, importance = TRUE, ntree = 500, mtry = 6)
## Look at variable importance:
round(importance(model_rf), 2)
importance(model_rf)
table(predict(model_rf), training$target)
Pred_rf = predict(model_rf, newdata=testing)
table(Pred_rf, testing$target)

cf_rf <- confusionMatrix(data=Pred_rf, reference=testing$target )
#cf_rf
summary(model_rf)
Accuracy_rf<-round(cf_rf$overall[1],2)
Accuracy_rf
precision(table(Pred_rf,testing$target))
recall(table(Pred_rf,testing$target))
precision <- posPredValue(Pred_rf, testing$target, positive="0")
precision
recall <- sensitivity(Pred_rf, testing$target, positive="0")
recall
F1_Score(testing$target, Pred_rf)
mean_absolute_error(testing$target, Pred_rf)
test_pred_rf <- predict(model_rf, test)
test['TARGETrf'] = as.data.frame(test_pred_rf)

#****************************************************************************************************
#FINAL PREDICTED TEST DATASET SAVED IN FILE PATH:
write.csv(test, file = "C:/Users/Gouri/Desktop/desktop/final_test_data.csv", row.names = FALSE)

#****************************************************************************************************