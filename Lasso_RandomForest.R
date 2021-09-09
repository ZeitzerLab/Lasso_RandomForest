rm(list=ls())

#Set directory
setwd("")
#read in data frame
df<-read.csv("")

#Select variables you want included in model
df2<-subset(df, select=c())

# Random Forest can only work with complte cases 
df2<- df2[complete.cases(df2),]

dim(df2)

## LASSO

#Choose seeds
set.seed(100) 

index = sample(1:nrow(df2), 0.7*nrow(df2)) 

train = df2[index,] # Create the training data 
test = df2[-index,] # Create the test data
dim(train)
dim(test)

#scale and Preprocess the selected variables 

cols = c()
pre_proc_val <- preProcess(train[,cols], method = c("center", "scale"))
train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])

summary(train)

cols_reg = c()
dummies <- dummyVars(Variable ~ ., data = df2[,cols_reg])
train_dummies = predict(dummies, newdata = train[,cols_reg])
test_dummies = predict(dummies, newdata = test[,cols_reg])

print(dim(train_dummies)); print(dim(test_dummies))

library(glmnet)

#Calculate lambda
x = as.matrix(train_dummies)
y_train = train$Variable

x_test = as.matrix(test_dummies)
y_test = test$Variable

lambdas <- 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(x, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = lambdas)

summary(ridge_reg)

cv_ridge <- cv.glmnet(x, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda

eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Prediction and evaluation on train data
predictions_train <- predict(ridge_reg, s = optimal_lambda, newx = x)
eval_results(y_train, predictions_train, train)

# Prediction and evaluation on test data
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = x_test)
eval_results(y_test, predictions_test, test)

##
lambdas <- 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements lasso regression
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

# Best 
lambda_best <- lasso_reg$lambda.min 
lambda_best

lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)

predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
eval_results(y_train, predictions_train, train)

predictions_test <- predict(lasso_model, s = lambda_best, newx = x_test)
eval_results(y_test, predictions_test, test)

#####

XM = model.matrix(variable~., data = df2)

set.seed(71)
cv.glmmodM1 <- cv.glmnet(XM,df2$variable,alpha=1)
cv.glmmodM1$lambda.min
cv.glmmodM1$lambda.1se
plot(cv.glmmodM1)
coef(cv.glmmodM1, s = "lambda.1se")
best_lambdaM1 <- cv.glmmodM1$lambda.1se
glm_best_lambdaM1 = glmnet (XM, df2$variable, family="gaussian", alpha = 1, lambda =best_lambdaM1)
glm_best_lambdaM1

set.seed(71)
M1lars <- lars(XM,df2$variable)
print(M1lars)

#### Random forest
# choose trees 
library(randomForest)
set.seed(71)
rf <-randomForest(variable~.,data=df2, ntree=500) 
print(rf)

#Select mtry value with minimum Out of Bag (OOB) error
mtry <- tuneRF(df2,df2$variable, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

#Build model using best mtry value 
set.seed(71)
rf <-randomForest(variable~.,data=df2, mtry=best.m, importance=TRUE,ntree=500)
print(rf)

#Higher the value of mean decrease accuracy, higher importance of the variable in the model.
#Mean Decrease Accuracy - How much the model accuracy decreases if we drop that variable.
importance(rf)
varImpPlot(rf)

library("rfUtilities")
rf.significance(rf, df2[,1:4], q = 0.99, p = 0.05, nperm = 999)
