# Prediction model
rm(list=ls())

setwd("C:/Users/valer/Desktop/University/Stockholm School of Economics/Courses/2nd year/Data Science Analytics/Prediction model")

# 0) Install packages, get the data
install.packages("plotROC")
install.packages("dlstats")
install.packages("pkgsearch")
install.packages("pROC")
install.packages("ggpubr")
install.packages("reshape")
install.packages("rfUtilities")

library(rfUtilities)
library(reshape)
library(ggpubr)
library(pROC)
library(dlstats)    # for package download stats
library(pkgsearch)
library(plotROC)
library(RMySQL)
library(dplyr)
library(tidyverse)
library(officer)
library(flextable)
library(huxtable)
library(jtools)
library ("readr")
library("dplyr")
library("ggplot2")
library("ggthemes")
library("tidyverse")
library("rio")
library("bit64")
library("psych")
library("tidyr")
library("caret")
library("e1071")
library("MASS")
library("rpart")
library("randomForest")
library("tree")
library("RMySQL")
library("lubridate")

# Retrieve dataset (company name is *****)
con = dbConnect(MySQL(), dbname = "*****",
                host = "db.cfda.se", port = 3306,
                user = "toystorey", password = "toys@sse")    # Insert real company name
dbListTables(con)

rs = dbSendQuery(con, "select * from *****AR")                # Insert real company name
compdat = fetch(rs, n=-1)

rs2 = dbSendQuery(con, "select * from *****Order")            # Insert real company name
orderds = fetch(rs2, n=-1)

# 1) Preparing the dataset

# Change the merchant_id to "Company 1", "Company 2"...:
orderds <- orderds %>% mutate(merchant_id = case_when(
  merchant_id == 258643 ~ "Company 1",
  merchant_id == 2026296 ~ "Company 2",
  merchant_id == 2723490 ~ "Company 3",
  merchant_id == 2756123 ~ "Company 4",
  merchant_id == 3642550 ~ "Company 5",
  merchant_id == 4218266 ~ "Company 6",
  merchant_id == 5258736 ~ "Company 7",
  merchant_id == 5314105 ~ "Company 8",
  merchant_id == 5913810 ~ "Company 9",
  merchant_id == 6394740 ~ "Company 10",
  merchant_id == 8244704 ~ "Company 11",
  merchant_id == 9402067 ~ "Company 12"
))

# I want to predict target as a function of the other variables
# Target: =1 if device is desktop, else =0
table(orderds$target)
table(orderds$device)

# Store only null values here
Null <- orderds[is.na(orderds$target),]

# create a database without null values where I will train and test the model
orderds2 <- subset(orderds, orderds$target == 1 | orderds$target == 0)
orderds2$target <- as.factor(orderds2$target)


# Transform variables in the correct type
orderds2$birthyear <- as.numeric(orderds2$birthyear)          #Make birthyear a numeric variable
orderds2$purchase_amount <- as.numeric(orderds2$purchase_amount)
orderds2$datestamp <- as.Date(orderds2$datestamp)
orderds2$target <- as.factor(orderds2$target)
orderds2$country <- as.factor(orderds2$country)
orderds2$gender <- as.factor(orderds2$gender)

# Create a variable for age 
orderds2$year <- year(orderds2$datestamp)
orderds2 <- orderds2 %>% mutate(age = year-birthyear)

# Create dummies for the various devices
orderds2$desktop <- ifelse(orderds2$device=="Desktop",1,0)
orderds2$tablet <- ifelse(orderds2$device=="Tablet",1,0)
orderds2$mobile <- ifelse(orderds2$device=="Mobile",1,0)
orderds2$other <- ifelse(orderds2$device=="Other",1,0)
orderds2$gameconsole <- ifelse(orderds2$device=="Game console",1,0)
orderds2$smarttv <- ifelse(orderds2$device=="Smart TV",1,0)
orderds2$empty <- ifelse(orderds2$device=="empty",1,0)


# Create the variables "month", "weekday", "day", "firstyear"
orderds2 <- orderds2 %>% mutate(month = month(datestamp))
orderds2 <- orderds2 %>% mutate(weekday = wday(datestamp))
orderds2 <- orderds2 %>% mutate(day = day(datestamp))
orderds2 <- orderds2 %>% mutate(firstyear = ifelse(year(datestamp)==2014,1,0))      
# This variable is a dummy which takes a value of 1 if the purchase was in the first year


# 2) Studying the dataset

# Seasonalities
hist(orderds2$month)    
# We see that the last months of the year have the highest number of purchases, especially December.

# Desktop by company
table(orderds$merchant_id)                                  # Companies 1, 3 and 5 have the most transactions
tapply(orderds2$desktop, orderds2$merchant_id, mean)
# Customers from companies 4, 6, 9, 10, 12 do not use the desktop. This could be because they have few 
# transactions.

# Desktop by country
table(orderds$country) 
tapply(orderds2$desktop, orderds2$country, mean)            # Norwegian customers use desktops less

# Desktop by gender
table(orderds$gender)                                       # Female customers are 56% of the sample, males only
                                                            # 29%
DgenderD0 <- table(orderds2$desktop,orderds2$gender)[1,]/sum(orderds2$desktop==0)
DgenderD1 <- table(orderds2$desktop,orderds2$gender)[2,]/sum(orderds2$desktop==1)
DgenderM <-tapply(orderds2$desktop, orderds2$gender, mean)
GENDER <- t(cbind(DgenderD1,DgenderD0,DgenderM))
rownames(GENDER) <- c( "Desktop","Other devices","Total")
colnames(GENDER) <- c( "Female","Male","Total")
GENDER

# Desktop by age
describe(orderds2$age)                                      # The mean is 43.42
ggdensity(orderds2, x = "age", 
          fill = "#0073C2FF", color = "#0073C2FF",
          add = "mean", rug = TRUE)

# Purchase amount
DDesk <- describe(orderds2$purchase_amount)
DDesk1 <- describe(orderds2[orderds2$desktop==1,]$purchase_amount)
DDesk0 <- describe(orderds2[orderds2$desktop==0,]$purchase_amount)
DDesk1
DDesk0
# The average purchase amount for customers that use the desktop is lower. The standard deviation and 
# especially the range are also much lower. This suggests that there are large outliers in desktop = 0.
plot(orderds2$purchase_amount) 
# We can see from the graph that while most purchases are below 1e+07, some outliers reach very high values.

Plot1 <- ggplot(orderds2, aes(x=purchase_amount, color=as.factor(desktop)))+
             geom_histogram(aes(y = (..count..)/sum(..count..)),binwidth = 10000,
                 fill="white",alpha=0.5, position="identity")+
              xlim(0,250000)+ 
            geom_vline(xintercept=DDesk$"mean", linetype="dotted") + 
            geom_text(x=DDesk$"mean"+1000, y=.1, label="Mean") +
            geom_vline(xintercept=DDesk$"median", linetype="dotted") + 
            geom_text(x=DDesk$"median"+1000, y=.1, label="Median")
Plot1                                  # Only works when you run Plot1" the second time, but the graph works.
# the distribution is left-skewed. The median is below the mean.


# Correlations
# A= corr >.05, B= corr >.1, C= corr >.15                               # Month
cor(orderds2$desktop,month(orderds2$datestamp)=="1") # A                # Negative correlation.
cor(orderds2$desktop,month(orderds2$datestamp)=="2") # A                # Negative correlation.
cor(orderds2$desktop,month(orderds2$datestamp)=="3") # A                # Negative correlation.
cor(orderds2$desktop,month(orderds2$datestamp)=="4") # A                # Negative correlation.
cor(orderds2$desktop,month(orderds2$datestamp)=="5") 
cor(orderds2$desktop,month(orderds2$datestamp)=="6") 
cor(orderds2$desktop,month(orderds2$datestamp)=="7") 
cor(orderds2$desktop,month(orderds2$datestamp)=="8")
cor(orderds2$desktop,month(orderds2$datestamp)=="9")
cor(orderds2$desktop,month(orderds2$datestamp)=="10")
cor(orderds2$desktop,month(orderds2$datestamp)=="11")
cor(orderds2$desktop,month(orderds2$datestamp)=="12") # A
                                                                        # Country
cor(orderds2$desktop,orderds2$country=="se")
cor(orderds2$desktop,orderds2$country=="no") # B                        # Negative correlation.
cor(orderds2$desktop,orderds2$country=="fi")
                                                                        # Gender
cor(orderds2$desktop,orderds2$gender=="male") # C
cor(orderds2$desktop,orderds2$gender=="female") # B
cor(orderds2$desktop,orderds2$gender=="none") # C                       # Negative correlation.

cor(orderds2$desktop,orderds2$weekday==1)
cor(orderds2$desktop,orderds2$weekday==2)
cor(orderds2$desktop,orderds2$weekday==3)
cor(orderds2$desktop,orderds2$weekday==4)
cor(orderds2$desktop,orderds2$weekday==5)
cor(orderds2$desktop,orderds2$weekday==6)
cor(orderds2$desktop,orderds2$weekday==7)

                                                                        # Purchase Amount
cor(orderds2$desktop,orderds2$purchase_amount) # A                      # Negative correlation.


# 3) Modelling

# Divide in training set and test set
set.seed(7313)
inTrain = createDataPartition(orderds2$target, p=0.5, list = F)
train = orderds2[inTrain,]
test = orderds2[-inTrain,]

# Because my pc is unable to handle properly such a large dataset, I will be executing directly a 
# Random Forest with a limited number of trees (10). In normal circumstances, a preliminary specification 
# with a logistic regression, lda, qda... would be advisable (see Assignment 5, where I used 5 preliminary 
# models on a smaller database).

# We now create a Random Forest
set.seed(7313)
# Random forest cannot handle missing data. We have to remove it
# Substitute with the median for numerical variables, with the mode for categorical variables
train$age <- ifelse(is.na(train$age), median(train$age, na.rm = TRUE), train$age)
train$month <- ifelse(is.na(train$month), mode(train$month, na.rm = TRUE), train$month)
test$age <- ifelse(is.na(test$age), median(test$age, na.rm = TRUE), test$age)
test$month <- ifelse(is.na(test$month), mode(test$month, na.rm = TRUE), test$month)

#Check there are no missing values
sum(is.na(train$target))
sum(is.na(train$purchase_amount)) 
sum(is.na(train$age))
sum(is.na(train$month))      
sum(is.na(train$gender))
sum(is.na(train$merchant_id)) 
sum(is.na(train$country)) 
sum(is.na(train$tablet)) 
sum(is.na(train$mobile)) 
sum(is.na(train$other)) 
sum(is.na(train$gameconsole)) 
sum(is.na(train$smarttv)) 

# Run the models
# Model 1: Univariate random forest
rf.fit <- randomForest(train$target~purchase_amount, data=train, ntree =10)    # ntree=10 or my pc can't run the model
pred = predict(rf.fit, train)
acc.tree.train = confusionMatrix(pred, train$target)
acc.tree.train # Accuracy is 59%, kappa is 0.058. The model is not very good, we should add age
# Test accuracy:
acc.tree.test = confusionMatrix(predict(rf.fit, test), test$target)
acc.tree.test # Accuracy is 58%, kappa is 0.030.

# Model 2: Numeric random forest
rf.fit2 <- randomForest(train$target~purchase_amount + age, data=train, ntree =10)
pred2 = predict(rf.fit2, train)
acc.tree.train2 = confusionMatrix(pred2, train$target)
acc.tree.train2 # Accuracy is 65%%, kappa is 0.27. Age is an important determinant. Now add other variables 
# Test accuracy:
acc.tree.test2 = confusionMatrix(predict(rf.fit2, test), test$target)
acc.tree.test2 # Accuracy is 63%, kappa is 0.022.

# Model 3: Numeric and categorical random forest
rf.fit3 <- randomForest(train$target~purchase_amount + age + month + weekday + country + gender, data=train,
                        ntree =10)
pred3 = predict(rf.fit3, train)
acc.tree.train3 = confusionMatrix(pred3, train$target)
acc.tree.train3 # Accuracy is 72%, kappa is 0.43. 
# Test accuracy:
acc.tree.test3 = confusionMatrix(predict(rf.fit3, test), test$target)
acc.tree.test3 # Accuracy is 69%, kappa is 0.35. The model's accuracy is good. 
# Given that in all three models the accuracy does not fall substantially from the training test to the test
# set, this suggests the data is not being overfitted.

# Model 4: Random forest with device dummies
rf.fit4 <- randomForest(train$target~purchase_amount + age + month + weekday + country + gender+ tablet +
                          mobile + other + gameconsole + smarttv, data=train, ntree =10)
# "Desktop" and "target" are the same variable, so I have not added it as a feature.
pred4 = predict(rf.fit4, train)
acc.tree.train4 = confusionMatrix(pred4, train$target)
acc.tree.train4 # Accuracy is 81%, kappa is 0.62
# Test accuracy:
acc.tree.test4 = confusionMatrix(predict(rf.fit4, test), test$target)
acc.tree.test4 # Accuracy is 81.5%, kappa is 0.61. 
# The accuracy of the model is even better, but the reason is that the device dummies are included as features.
# These will not be available in the database "Null", and thus this model can't be used for predictions.

# Let's now examine the importance of our features for Model 3
importance(rf.fit3)
# The gini coefficients say how pure the nodes are (purer is better). The gini coeff takes lower values with
# purer nodes.
# As we add a variable the coefficient decreases
varImpPlot(rf.fit3)

# Specificity and sensitivity of my main model
predROC = predict(rf.fit3, train, type = "prob")
rf.ROC <- roc(train$target, rf.fit3$votes[,2])
plot(rf.ROC)
print(rf.ROC)             # The Area Under the Curve (AUC) is 70%
# The model has a high sensitivity (true positive rate) and specificity (true negative rate). We can calculate 
# this more precisely by looking at the confusion matrix: 76% of true negative predictions are correct 
# (208558/272379) and 67% (127091/187568) of true positive predictions are correct.
acc.tree.train3

# If I had made preliminary models with a logistic regression / lda / qda, I would compare the results using 
# cross-validation (CV):
control = trainControl(method = "cv", number = 5, 
                       classProbs = T, summaryFunction = twoClassSummary) 
results = resamples(list(
  logit =fit.glm, lda = fit.lda, qda = fit.qda))
dotplot(results)
# Some claim that CV with Random Forest can be redundant, and since in this particular case we would be comparing
# similar models with the slightly different features, I will skip it.


# 4) Predictions
# Make the same transformations in Null that were made in orderds2
# We will use Model 3

Null <- Null %>% mutate(merchant_id = case_when(
  merchant_id == 258643 ~ "Company 1",
  merchant_id == 2026296 ~ "Company 2",
  merchant_id == 2723490 ~ "Company 3",
  merchant_id == 2756123 ~ "Company 4",
  merchant_id == 3642550 ~ "Company 5",
  merchant_id == 4218266 ~ "Company 6",
  merchant_id == 5258736 ~ "Company 7",
  merchant_id == 5314105 ~ "Company 8",
  merchant_id == 5913810 ~ "Company 9",
  merchant_id == 6394740 ~ "Company 10",
  merchant_id == 8244704 ~ "Company 11",
  merchant_id == 9402067 ~ "Company 12"
))


# Transform variables in the correct type
Null$target <- as.factor(Null$target)
Null$birthyear <- as.numeric(Null$birthyear)          #Make birthyear a numeric variable
Null$purchase_amount <- as.numeric(Null$purchase_amount)
Null$datestamp <- as.Date(Null$datestamp)
Null$target <- as.factor(Null$target)
Null$country <- as.factor(Null$country)
Null$gender <- as.factor(Null$gender)

# Create a variable for age 
Null$year <- year(Null$datestamp)
Null <- Null %>% mutate(age = year-birthyear)

# Create dummies for the various devices
Null$desktop <- ifelse(Null$device=="Desktop",1,0)
Null$tablet <- ifelse(Null$device=="Tablet",1,0)
Null$mobile <- ifelse(Null$device=="Mobile",1,0)
Null$other <- ifelse(Null$device=="Other",1,0)
Null$gameconsole <- ifelse(Null$device=="Game console",1,0)
Null$smarttv <- ifelse(Null$device=="Smart TV",1,0)
Null$empty <- ifelse(Null$device=="empty",1,0)

#Create the variables "month", "weekday", "day", "firstyear"
Null <- Null %>% mutate(month = month(datestamp))
Null <- Null %>% mutate(weekday = wday(datestamp))
Null <- Null %>% mutate(day = day(datestamp))
Null <- Null %>% mutate(firstyear = ifelse(year(datestamp)==2014,1,0))      
# This variable is a dummy which takes a value of 1 if the purchase was in the first year

Null$age <- ifelse(is.na(Null$age), median(Null$age, na.rm = TRUE), Null$age)
Null$month <- ifelse(is.na(Null$month), mode(Null$month, na.rm = TRUE), Null$month)

predFinal
predFinal = predict(rf.fit3, Null)
Null$pred <- predFinal
Predictions <- Predictions[-2]
Predictions <- Null[,-5]
help("subset")
write.csv(Predictions, "PredictionsFinal")                   # The code obtained is then imported in an xls
# in order to separate it in the columns "Id" and "target" as requested in the assignment.


# Appendix Code

# I would like to improve accuracy above 70%, so I add a PCA
set.seed(7313)

# The PCA requires that we do not have categorical data, but only numeric.
orderds3 <- orderds2[, 13:22]
orderds3 <- orderds3[-2]
orderds3 <- orderds3[-7]
orderds3$purchase_amount <- orderds2$purchase_amount
# Eliminate missing values
orderds3$age <- ifelse(is.na(orderds3$age), median(orderds3$age, na.rm = TRUE), orderds3$age)
orderds3$month <- ifelse(is.na(orderds3$month), mode(orderds3$month, na.rm = TRUE), orderds3$month)

# Run PCA
pca.fit <- prcomp(orderds3, scale. = TRUE)
summary(pca.fit)              # There are 9 principal components, like our 9 variables
pca.fit$rotation              # Rotation:  relationship between the initial variables and the principal components
# calculate and plot percentage of variance explained
pve = pca.fit$sdev^2 / sum(pca.fit$sdev^2)
plot(pve)
plot(cumsum(pve), type = "b") # All principal components explain a similar proportion of the variance (11%).
# The pca does not seem to add value to the analysis, especially as it would also make our model results 
# harder to interpret. We thus do not add it to Model 3