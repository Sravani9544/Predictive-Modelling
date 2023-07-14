## Load the necessary libraries (after making sure that they are installed)
## devtools::install_github("ModelOriented/drifter")
library(drifter) 
library(caret)
library(doParallel)
library(randomForest)
library(glmnet)

## Load the dataset used in Exam 1
dataset <- read.csv("dataset.csv")
endrow <- nrow(dataset) * 0.9
ds1 <- dataset[1:endrow, ]
ds2 <- dataset[(endrow+1): nrow(dataset), ]

summary(ds1)
summary(ds2)
## See if the two subsets are different from each other
calculate_covariate_drift(ds1, ds2) ## they are! Look at the distance values

## Next, fit some models using CARET
## Pre-processing
preProcVals1 <- preProcess(ds1, method=c("center", "scale"))
preProcVals2 <- preProcess(ds2, method=c("center", "scale"))

## Set up for training
fitControl <- trainControl(
  method = "repeatedcv", ## perform repeated k-fold CV
  number = 5,
  repeats = 1)

## Set up for parallelized code execution
cores <- detectCores()
cl <- makePSOCKcluster(cores) #or cores-1
registerDoParallel(cl)

## Lasso
lasso1 <- train(Y ~ . - X,                          ##
               data = ds1,                          ##
               method = "glmnet",                   ##
               trControl = fitControl,              ##
               verbose = FALSE)                     ##
                                                    ##
lasso2 <- train(Y ~ . - X,                          ##
               data = ds2,                          ##
               method = "glmnet",                   ##
               trControl = fitControl,              ##
               verbose = FALSE)                     ##
                                                    ##
## check drift                                      ##
check_drift(lasso1, lasso2, ds1, ds2, ds1$Y, ds2$Y)

## Fit random forest models using the CARET interface
grid <- expand.grid(mtry = 1:3)#(ncol(ds1)-1))

forestfit1 <- train(Y ~ . - X,
                   data = ds1,
                   method = "rf",
                   trControl = fitControl,
                   verbose = FALSE,
                   tuneGrid = grid)

forestfit2 <- train(Y ~ . -X,
                   data = ds2,
                   method = "rf",
                   trControl = fitControl,
                   verbose = FALSE,
                   tuneGrid = grid)

## check drift in models - whether the models built on these models behave differently in predicting Y
check_drift(forestfit1, forestfit2, ds1, ds2, ds1$Y, ds2$Y)

## Fit using native function, directly using the native randomForest() function
rf1 <- randomForest(ds1[, 2:5], ds1$Y)
rf2 <- randomForest(ds2[, 2:5], ds2$Y)

## Check model drift
check_drift(rf1, rf2, ds1, ds2, ds1$Y, ds2$Y)

