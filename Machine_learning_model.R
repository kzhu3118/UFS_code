library(caret)
library(pROC)
library(kknn)
library(openxlsx)
library(ggplot2)
library(reshape2)
library(e1071)
library(skimr)
library(DataExplorer)
library(tidyverse)
library(kernlab)
library(xgboost)
library(Matrix)
library(raster)
library(randomForest)
library(ggRandomForests)
library(vivid)
library(MASS)
library(network)
library(sna)
library(intergraph)
library(lemon)
library(rpart)
library(rpart.plot)

# 1. KNN model --------------------------------------------------------------
## 1.1 Model building ------------------------------------------------------

# dataset
ft_knn <- read.xlsx("data/Machine_learning_model/rf.figure.xlsx")

ft_knn$Functional <- factor(ft_knn$Functional,
  levels = c("f1", "f2", "f3", "f4", "f5"),
  labels = c("CFA", "RFA", "IFA", "PFA", "TFA")
)

set.seed(666)
l <- caret::createDataPartition(ft_knn$Functional, p = 0.8, list = FALSE)
trainset <- ft_knn[l, ]
testset <- ft_knn[-l, ]
dim(trainset)
dim(testset)
trainset$Functional <- as.factor(trainset$Functional)
testset$Functional <- as.factor(testset$Functional)

# KNN model
set.seed(3333)
# Create an empty vector to store the cross-validation error rate
cv.error <- rep(0, 10)

# Perform cross-validation on each k value
for (k in 1:10) {
  ft_kknn <- kknn(Functional ~ ., train = trainset, test = testset, scale = TRUE, k = k)
  cv.error[k] <- sum(testset$Functional != fitted(ft_kknn))
}

# Visualize the cross-validation error rate graphically
plot(1:10, cv.error, type = "b", main = "Cross-validation error vs. k", xlab = "Number of Nearest Neighbors", ylab = "Error Rate")

# Find the k value with the lowest error rate
best.k <- which.min(cv.error)

# Modeling using the optimal k value
knnn_train <- kknn(Functional ~ ., train = trainset, test = testset, scale = TRUE, k = best.k)
pred <- fitted(knnn_train)

# Generate confusion matrix

confusionMatrix(testset$Functional, pred)

# Required libraries

# Predict probabilities using the model
prob <- predict(knnn_train, testset, type = "prob")

# roc results presentation
roc.rf <- multiclass.roc(testset$Functional, prob) # Don't use ROC directly; this is a multi-class classification problem, so don't use Plot directly.
# Because this involves multiple classes, we need to select one class to calculate.
head(roc.rf)


## 1.2 ROC and AUC ---------------------------------------------------------

## Drawing
# A list to save ROC and AUC for each class
roc_list <- list()
auc_list <- numeric()

col_names <- c("CFA", "RFA", "IFA", "PFA", "TFA")

# For each class, generate the ROC curve
for (i in 1:5) {
  # Compute ROC and AUC for each class
  roc_res <- roc(ifelse(testset$Functional == col_names[i], 1, 0), prob[, i])

  # Save the result
  roc_list[[i]] <- roc_res
  auc_list[i] <- auc(roc_res)
}

colors <- c("#8A2BE2", "#FC4E07", "#E7B800", "#00AFBB", "#5F9EA0")

# plot ROC curve
plot(roc_list[[1]], col = colors[1])
for (i in 2:5) {
  lines(roc_list[[i]], col = colors[i])
}

# Add a legend
legend("bottomright", legend = paste(col_names, ":", round(auc_list, 2)), lty = 1, col = colors, bty = "n")


## 1.3 Confusion matrix plotting -------------------------------------------

result <- confusionMatrix(testset$Functional, pred)
cm <- result$table

# Convert the confusion matrix to a data frame
cm_df <- as.data.frame(cm)

# Use the ggplot2 and reshape2 packages to convert data frame formats and plot heatmaps
# The melt function is used to convert a wide data frame to a long data frame.
cm_long <- melt(cm_df)

# Drawing heatmaps
ggplot(data = cm_long, aes(x = Prediction, y = Reference, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "black") + # Add quantity value
  theme_minimal() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"), # Set the x-axis text color to black.
    axis.text.y = element_text(color = "black"), # Set the y-axis text color to black.
    axis.line = element_line(colour = "black"), # Set the axis color to black.
    panel.border = element_rect(colour = "black", fill = NA, size = 1) # Set the panel border to black.
  ) +
  labs(x = "Predicted", y = "Actual", fill = "")

# 2. SVM model ---------------------------------------------------------------
## 2.1 Model building ----------------------------------------------------------

ft_svm <- read.xlsx("data/Machine_learning_model/rf.figure.xlsx")

ft_svm$Functional <- factor(ft_svm$Functional,
  levels = c("f1", "f2", "f3", "f4", "f5"),
  labels = c("CFA", "RFA", "IFA", "PFA", "TFA")
)

skim(ft_svm)
ft_svm$Functional <- as.factor(ft_svm$Functional) # Convert to factor data
table(ft_svm$Functional)

summary(ft_svm)

# Split data
set.seed(666)
trains <- createDataPartition(y = ft_svm$Functional, p = 0.8, list = F) # Instead of returning a list, return a data frame.
traindata <- ft_svm[trains, ]
testdata <- ft_svm[-trains, ]
dim(traindata)
dim(testdata)

# Examine the distribution of the dependent variable after splitting.
table(traindata$Functional)
table(testdata$Functional)

# Set control parameters for cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# Formula for constructing dependent and independent variables
colnames(ft_svm)
form_clsm <- as.formula(
  paste0("Functional~", paste(colnames(traindata)[2:8], collapse = "+"))
)
form_clsm

# Building Model
set.seed(666)

fit_svm_clsm <- train(form_clsm,
  data = traindata,
  method = "svmRadial",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 10
)

# Output model information and view model parameters
print(fit_svm_clsm)

# Output optimal parameters
print(fit_svm_clsm$bestTune)

# Predict the probability on the test set using the optimal model.
testpred <- predict(fit_svm_clsm, newdata = testdata, probability = T) # probability

fit_svm_clsm <- svm(Functional ~ ., data = testdata, probability = TRUE)
prob <- predict(fit_svm_clsm, newdata = testdata, probability = T)
prob_values <- attr(prob, "probabilities")
head(prob_values)


# Test Set ROC
# Extracting Predicted Probabilities

testpredprob <- attr(testpred, "probabilities") # Extracting probability components
roc.rf <- multiclass.roc(as.numeric(testdata$Functional), as.numeric(testpred))
roc.rf

# Test Set Obfuscation Matrix
confusionMatrix(data = testpred, reference = testdata$Functional, mode = "everything")

# Test Set Overall Results
multiClassSummary(
  data.frame(obs = testdata$Functional, pred = testpred),
  lev = levels(testdata$Functional)
)


## 2.2  ROC and AUC ----------------------------------------------

roc_list <- list()
auc_list <- numeric()

col_names <- c("f1", "f2", "f3", "f4", "f5")
col_names <- c("CFA", "RFA", "IFA", "PFA", "TFA")


# For each class, generate the ROC curve
for (i in 1:5) {
  # Compute ROC and AUC for each class
  roc_res <- roc(ifelse(testdata$Functional == col_names[i], 1, 0), prob_values[, i])

  # Save the result
  roc_list[[i]] <- roc_res
  auc_list[i] <- auc(roc_res)
}

colors <- c("#8A2BE2", "#FC4E07", "#E7B800", "#00AFBB", "#5F9EA0")

# plot ROC curve
plot(roc_list[[1]], col = colors[1])
for (i in 2:5) {
  lines(roc_list[[i]], col = colors[i])
}

# Add a legend
legend("bottomright", legend = paste(col_names, ":", round(auc_list, 2)), lty = 1, col = colors, bty = "n")


## 2.3 Confusion matrix plotting -----------------------------------------------

# Confusion matrix：plot_zoom_png?width=389&height=324
result <- confusionMatrix(data = testpred, reference = testdata$Functional, mode = "everything")
cm <- result$table

# Convert the confusion matrix to a data frame
cm_df <- as.data.frame(cm)

# Use the ggplot2 and reshape2 packages to convert data frame formats and plot heatmaps


# The melt function is used to convert a wide data frame to a long data frame.
cm_long <- melt(cm_df)

# Drawing heatmaps
ggplot(data = cm_long, aes(x = Prediction, y = Reference, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "black") +
  theme_minimal() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.line = element_line(colour = "black"),
    panel.border = element_rect(colour = "black", fill = NA, size = 1)
  ) +
  labs(x = "Predicted", y = "Actual", fill = "")

# 3. XGBoost model -----------------------------------------------------------

## 3.1 Model building ------------------------------------------------------
ft_xgb <- read.xlsx("data/Machine_learning_model/rf.figure.xlsx")

ft_xgb$Functional <- factor(ft_xgb$Functional,
  levels = c("f1", "f2", "f3", "f4", "f5"),
  labels = c("CFA", "RFA", "IFA", "PFA", "TFA")
)

skim(ft_xgb)
plot_missing(ft_xgb)
# DataExplorer::set_missing() can fill in missing data

# Variable type correction
for (i in c(1)) {
  ft_xgb[, i] <- factor(ft_xgb[, i])
}

# Bird's-eye view after data processing
skim(ft_xgb)
table(ft_xgb$Functional)

# Split data
set.seed(666)
trainlist <- createDataPartition(y = ft_xgb$Functional, p = 0.8, list = F)
trainset <- ft_xgb[trainlist, ]
testset <- ft_xgb[-trainlist, ]
dim(trainset)
dim(testset)

# Data Preparation
## Training Set Data Preprocessing
traindata1 <- data.matrix(trainset[, 2:8])
traindata2 <- Matrix(traindata1, sparse = T)
train_y <- as.numeric(trainset$Functional) - 1
traindata <- list(data = traindata2, label = train_y)
dtrain <- xgb.DMatrix(data = traindata$data, label = traindata$label)

## Test Set Data Preprocessing
testdata1 <- data.matrix(testset[, 2:8])
testdata2 <- Matrix(testdata1, sparse = T)
test_y <- as.numeric(testset$Functional) - 1
testdata <- list(data = testdata2, label = test_y)
dtest <- xgb.DMatrix(data = testdata$data, label = testdata$label)

watchlist <- list(train = dtrain, test = dtest)

# Model Training

# Parameter Settings
fit_xgb_cls <- xgb.train(
  data = dtrain,
  eta = 0.3,
  gamma = 0.001,
  max_depth = 2,
  subsample = 0.7,
  colsample_bytree = 0.4,
  num_class = 5,
  objective = "multi:softprob",
  nrounds = 751,
  watchlist = watchlist,
  verbose = 1,
  print_every_n = 100,
  early_stopping_rounds = 200
)

# Output model information and view model parameters
print(fit_xgb_cls)

# Variable Importance
importance_matrix <- xgb.importance(model = fit_xgb_cls)
print(importance_matrix)
xgb.plot.importance(
  importance_matrix = importance_matrix,
  measure = "Cover",
  col = "skyblue",
  cex = 1,
  main = "XGBoost变量重要性"
)
# SHAP
xgb.plot.shap(
  data = traindata1,
  model = fit_xgb_cls,
  las = 1,
  top_n = 5,
  pch = 5
)

## Predicting probabilities on the test set
testpredprob <- predict(fit_xgb_cls, newdata = dtest)
testpredprob2 <- as.data.frame(matrix(testpredprob, ncol = 5, byrow = T))
colnames(testpredprob2) <- c("CFA", "RFA", "IFA", "PFA", "TFA")

# Test set prediction classification
testpredprob3 <- testpredprob2
testpredprob3$lab <- as.factor(
  apply(testpredprob2, 1, which.max)
)

# Test Set ROC
multiclass.roc(
  response = testset$Functional,
  predictor = testpredprob2
)

# Training set confusion matrix
testpredprob3$lab2 <- factor(testpredprob3$lab, levels = c(1, 2, 3, 4, 5), labels = c("CFA", "RFA", "IFA", "PFA", "TFA"))

confusionMatrix(
  data = testpredprob3$lab2,
  reference = testset$Functional,
  mode = "everything"
)
# Test Set Overall Results
multiClassSummary(
  data.frame(
    obs = testset$Functional, # Categories observed and predicted
    pred = testpredprob3$lab2
  ),
  lev = levels(testset$Functional)
)


## 3.2 ROC and AUC -------------------------------------------------------------
roc_list <- list()
auc_list <- numeric()

col_names <- c("CFA", "RFA", "IFA", "PFA", "TFA")

# For each class, generate the ROC curve
for (i in 1:5) {
  # Compute ROC and AUC for each class
  roc_res <- roc(ifelse(testset$Functional == col_names[i], 1, 0), testpredprob2[, i])

  # Save the result
  roc_list[[i]] <- roc_res
  auc_list[i] <- auc(roc_res)
}

colors <- c("#8A2BE2", "#FC4E07", "#E7B800", "#00AFBB", "#5F9EA0")

# plot ROC curve
plot(roc_list[[1]], col = colors[1])
for (i in 2:5) {
  lines(roc_list[[i]], col = colors[i])
}

# Add a legend
legend("bottomright", legend = paste(col_names, ":", round(auc_list, 2)), lty = 1, col = colors, bty = "n")


## 3.3 Confusion matrix plotting -------------------------------------------
result <- confusionMatrix(
  data = testpredprob3$lab2,
  reference = testset$Functional,
  mode = "everything"
)
cm <- result$table

# Convert the confusion matrix to a data frame
cm_df <- as.data.frame(cm)

# Use the ggplot2 and reshape2 packages to convert data frame formats and plot heatmaps
cm_long <- melt(cm_df)

# Drawing heatmaps
ggplot(data = cm_long, aes(x = Prediction, y = Reference, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "black") +
  theme_minimal() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"),
    axis.line = element_line(colour = "black"),
    panel.border = element_rect(colour = "black", fill = NA, size = 1)
  ) +
  labs(x = "Predicted", y = "Actual", fill = "")


# 4. Random Forest model -----------------------------------------------------


## 4.1 Model building -----------------------------------------------------
# database

ft_rf <- read.xlsx("data/Machine_learning_model/rf.figure.xlsx")

ft_rf$Functional <- factor(ft_rf$Functional,
  levels = c("f1", "f2", "f3", "f4", "f5"),
  labels = c("CFA", "RFA", "IFA", "PFA", "TFA")
)

# Divide the dataset into training and test sets
set.seed(666)
trainlist <- caret::createDataPartition(rf.0324$Functional, p = 0.8, list = FALSE) # Specify package, p represents the percentage of the whole.
# No need to specify the specific row number
trainset <- ft_rf[trainlist, ]
testset <- ft_rf[-trainlist, ]
dim(trainset)
dim(testset)

rf.train <- randomForest(as.factor(Functional) ~ .,
  data = trainset, importance = TRUE, ntree = 500, na.action = na.pass
)
rf.train
plot(rf.train, main = "raindomForest origin")

# Calculate the importance of variables
importance(rf.train)
# Draw a graph of variable importance
varImpPlot(rf.train)

# Fitting a regression model
# Converting text-based dependent variables to factor types
trainset$Functional <- factor(trainset$Functional)
rf.train2 <- tuneRF(trainset[, 2:ncol(trainset)], trainset[, "Functional"])
mt <- rf.train2[which.min(rf.train2[, 2]), 1]
mt

# Adjust the ntree function
rf.train <- randomForest(trainset[, 2:ncol(trainset)], trainset[, "Functional"], mtry = mt)
rf.train
plot(rf.train)

# Calculate the importance of variables
importance(rf.train)
# Draw a graph of variable importance
varImpPlot(rf.train)
gg_data <- gg_vimp(rf.train)
plot(gg_data)

# 2. predicting in testdata
set.seed(666)
rf.test <- predict(rf.train, newdata = testset, type = "class") # The type of thing to predict, so use class
rf.test

rf.cf <- caret::confusionMatrix(as.factor(rf.test), as.factor(testset$Functional))
rf.cf


## 4.2 ROC and AUC ---------------------------------------------------------

rf.test2 <- predict(rf.train, newdata = testset, type = "prob")
# You can't directly use `class` to draw an ROC curve; `prob` represents
# probability, which is very important; the probability distribution is 0-1 (the predicted probability of each flower in each category).
head(rf.test2)
roc.rf <- multiclass.roc(testset$Functional, rf.test2) # Don't use ROC directly, this is a multi-class classification problem, don't use Plot directly.
# Because this involves multiple classes, we need to select one class to calculate.
head(roc.rf)

## Drawing
# A list to save ROC and AUC for each class
roc_list <- list()
auc_list <- numeric()

col_names <- c("CFA", "RFA", "IFA", "PFA", "TFA")

# For each class, generate the ROC curve
for (i in 1:5) {
  # Compute ROC and AUC for each class
  roc_res <- roc(ifelse(testset$Functional == col_names[i], 1, 0), rf.test2[, i])

  # Save the result
  roc_list[[i]] <- roc_res
  auc_list[i] <- auc(roc_res)
}

colors <- c("#8A2BE2", "#FC4E07", "#E7B800", "#00AFBB", "#5F9EA0")

# plot ROC curve
plot(roc_list[[1]], col = colors[1])
for (i in 2:5) {
  lines(roc_list[[i]], col = colors[i])
}

# Add a legend
legend("bottomright", legend = paste(col_names, ":", round(auc_list, 2)), lty = 1, col = colors, bty = "n")


## 4.3 Confusion matrix plotting -------------------------------------------

result <- caret::confusionMatrix(as.factor(rf.test), as.factor(testset$Functional))
cm <- result$table

# Convert the confusion matrix to a data frame
cm_df <- as.data.frame(cm)

# Use the ggplot2 and reshape2 packages to convert data frame formats and plot heatmaps

# The melt function is used to convert a wide data frame to a long data frame.
cm_long <- melt(cm_df)

# Drawing heatmaps
ggplot(data = cm_long, aes(x = Prediction, y = Reference, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "black") +
  theme_minimal() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.line = element_line(colour = "black"),
    panel.border = element_rect(colour = "black", fill = NA, size = 1)
  ) +
  labs(x = "Predicted", y = "Actual", fill = "")


## 4.4 Variable importance and interaction ---------------------------------
set.seed(123)
VIVI_rf <- vivi(
  data = Boston,
  fit = rf_model,
  response = "medv"
)

# Expanding the previous code yields the same result.
set.seed(123)
VIVI_rf <- vivi(
  data = trainset,
  fit = rf.train,
  response = "Functional",
  reorder = TRUE,
  normalized = FALSE,
  importanceType = "agnostic",
  gridSize = 50,
  nmax = 500,
  class = 5,
  predictFun = NULL,
  numPerm = 4
)

print(VIVI_rf, digits = 1)

# Figure 1 shows the results plotted as a heatmap.
viviHeatmap(mat = VIVI_rf)

# Change heatmap color
viviHeatmap(
  mat = VIVI_rf,
  impPal = rev(colorspace::sequential_hcl(palette = "Reds 3", n = 100))
)

# Figure 2 Network Diagram

viviNetwork(mat = VIVI_rf)

# Set the threshold for the network graph
viviNetwork(mat = VIVI_rf, intThreshold = 0.2, removeNode = TRUE)

# Change the image style (square).
viviNetwork(
  mat = VIVI_rf,
  layout = igraph::layout_on_grid
)

# Figure 3 Partial Dependency Graph

pdpVars(
  data = trainset,
  fit = rf.train,
  response = "Functional",
  vars = colnames(VIVI_rf)[1:7],
  nIce = 100
)

# 5. Decision Tree -----------------------------------------------------

## 5.1 Model building ------------------------------------------------------

ft_dt <- read.xlsx("data/Machine_learning_model/rf.figure.xlsx")

ft_dt$Functional <- factor(ft_dt$Functional,
  levels = c("f1", "f2", "f3", "f4", "f5"),
  labels = c("CFA", "RFA", "IFA", "PFA", "TFA")
)

skim(ft_dt)
ft_dt$Functional <- as.factor(ft_dt$Functional) # Convert to factor data
table(ft_dt$Functional)

# Split data
set.seed(666)
trainlist <- caret::createDataPartition(ft_dt$Functional, p = 0.8, list = FALSE)
# Specify the package, p represents the percentage of the whole
# No need to specify the specific line number

trainset <- ft_dt[trainlist, ]
testset <- ft_dt[-trainlist, ]
dim(trainset)
dim(testset)

# Formula for constructing dependent and independent variables
colnames(ft_dt)
form_clsm <- as.formula(
  paste0("Functional~", paste(colnames(traindata)[2:8], collapse = "+"))
)
form_clsm

# Building a Model
fit_dt_clsm <- rpart(
  form_clsm,
  data = trainset,
  method = "class",
  parms = list(split = "information"), # Split Rules
  control = rpart.control(cp = 0.005) # Complexity parameter
)

# Original Data Classification Tree
fit_dt_clsm

# Complexity-related data
printcp(fit_dt_clsm)
plotcp(fit_dt_clsm)

# Post-pruning
fit_dt_clsm_pruned <- prune(fit_dt_clsm, cp = 0.0055)
print(fit_dt_clsm_pruned)

# Variable Importance
fit_dt_clsm_pruned$variable.importance
barplot(fit_dt_clsm_pruned$variable.importance)
# Graphing the Importance of Variables
varimpdata <- data.frame(importance = fit_dt_clsm_pruned$variable.importance)
ggplot(
  varimpdata,
  aes(x = as_factor(rownames(varimpdata)), y = importance)
) +
  geom_col() +
  labs(x = "variables") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1)) +
  theme_classic()

# Tree Diagram
# rpart.plot Reference Documentation
prp(fit_dt_clsm_pruned,
  type = 2,
  extra = 104,
  tweak = 4,
  fallen.leaves = TRUE,
  main = "Decision Tree"
)
# Create an RND document for further adjustments

# Perform post-pruning adjustments based on model performance

# Test set prediction probabilities

testpredprob <- predict(fit_dt_clsm_pruned, newdata = testset, type = "prob")
# Test Set ROC
multiclass.roc(response = testset$Functional, predictor = testpredprob)
# Test set prediction classification
testpredlab <- predict(fit_dt_clsm_pruned, newdata = testset, type = "class")
# Test Set Obfuscation Matrix
confusionMatrix(data = testpredlab, reference = testset$Functional, mode = "everything")
# Test Set Overall Results
multiClassSummary(
  data.frame(obs = testset$Functional, pred = testpredlab),
  lev = levels(testset$Functional)
)


## 5.2 ROC and AUC ---------------------------------------------------------

# A list to save ROC and AUC for each class
roc_list <- list()
auc_list <- numeric()

col_names <- c("CFA", "RFA", "IFA", "PFA", "TFA")

# For each class, generate the ROC curve
for (i in 1:5) {
  # Compute ROC and AUC for each class
  roc_res <- roc(ifelse(testset$Functional == col_names[i], 1, 0), testpredprob[, i])

  # Save the result
  roc_list[[i]] <- roc_res
  auc_list[i] <- auc(roc_res)
}

colors <- c("#8A2BE2", "#FC4E07", "#E7B800", "#00AFBB", "#5F9EA0")

# plot ROC curve
plot(roc_list[[1]], col = colors[1])
for (i in 2:5) {
  lines(roc_list[[i]], col = colors[i])
}

# Add a legend
legend("bottomright", legend = paste(col_names, ":", round(auc_list, 2)), lty = 1, col = colors, bty = "n")


## 5.3 Confusion matrix plotting -------------------------------------------

result <- confusionMatrix(data = testpredlab, reference = testset$Functional, mode = "everything")
cm <- result$table

# Convert the confusion matrix to a data frame
cm_df <- as.data.frame(cm)


cm_long <- melt(cm_df)

ggplot(data = cm_long, aes(x = Prediction, y = Reference, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "black") +
  theme_minimal() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.line = element_line(colour = "black"),
    panel.border = element_rect(colour = "black", fill = NA, size = 1)
  ) +
  labs(x = "Predicted", y = "Actual", fill = "")


# kfold_cv <- 5 # Sets the number of k-fold cross-validations

# Split data
set.seed(666)
trainlist <- caret::createDataPartition(ft_dt$Functional, p = 0.8, list = FALSE)
trainset <- ft_dt[trainlist, ]
testset <- ft_dt[-trainlist, ]
dim(trainset)
dim(testset)

# Formula for constructing dependent and independent variables

colnames(ft_dt)
form_clsm <- as.formula(
  paste0("Functional~", paste(colnames(trainset)[2:8], collapse = "+"))
)
form_clsm

# Building a Model
fit_dt_clsm <- rpart(
  form_clsm,
  data = trainset,
  method = "class",
  parms = list(split = "information"), # Split Rules
  control = rpart.control(cp = 0.005) # Complexity parameter
)

# Original Data Classification Tree
fit_dt_clsm

# Complexity-related data
printcp(fit_dt_clsm)
plotcp(fit_dt_clsm)

# Post-pruning
fit_dt_clsm_pruned <- prune(fit_dt_clsm, cp = 0.0055)
print(fit_dt_clsm_pruned)

# Variable Importance
fit_dt_clsm_pruned$variable.importance
barplot(fit_dt_clsm_pruned$variable.importance)
# Graphing the Importance of Variables
varimpdata <- data.frame(importance = fit_dt_clsm_pruned$variable.importance)
ggplot(
  varimpdata,
  aes(x = as_factor(rownames(varimpdata)), y = importance)
) +
  geom_col() +
  labs(x = "variables") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1)) +
  theme_classic()

# Tree Diagram
# rpart.plot Reference Documentation
prp(fit_dt_clsm_pruned,
  type = 2,
  extra = 104,
  tweak = 4,
  fallen.leaves = TRUE,
  main = "Decision Tree"
)

# k-fold cross-validation
for (i in 1:kfold_cv) {
  train_index <- createDataPartition(ft_dt$Functional, p = 0.8, list = FALSE) # Specify package, p represents the percentage of the whole.
  training_set <- rf.0324[train_index, ]
  test_set <- rf.0324[-train_index, ]
  form_clsm <- as.formula(
    paste0("Functional~", paste(colnames(training_set)[2:8], collapse = "+"))
  )
  fit_dt_cv <- rpart(
    form_clsm,
    data = training_set,
    method = "class",
    parms = list(split = "information"), # Split Rules
    control = rpart.control(cp = 0.005) # Complexity parameter
  )
  fit_dt_cv$predicted <- predict(fit_dt_cv, newdata = test_set, type = "class")
  acc_score[i] <- sum(fit_dt_cv$predicted == test_set$Functional) / nrow(test_set) * 100
}
acc_mean <- mean(acc_score)
acc_sd <- sd(acc_score)
cat("Accuracy:", round(acc_mean, 2), "% (", round(acc_sd, 2), "%)")
