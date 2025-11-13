library(sf)
library(sp)
library(raster)
library(foreign)
library(dplyr)
library(openxlsx)
library(foreign)
library(openxlsx)
library(raster)
library(caret)
library(pROC)
library(randomForest)
library(ggRandomForests)

# 1. Function area data preprocessing ----------------------------------------------------------------

# Load road network data
road_20 <- read_sf("data/UFS_classification_prediction/road_2020.gpkg")

Sumf1 <- read.dbf("Sum_f1.dbf") # Commercial service functional area
Sumf2 <- read.dbf("Sum_f2.dbf") # Residential functional area
Sumf3 <- read.dbf("Sum_f3.dbf") # Industrial functional area
Sumf4 <- read.dbf("Sum_f4.dbf") # Public service functional area
Sumf5 <- read.dbf("Sum_f5.dbf") # Transportation service functional area
Sumf6 <- read.dbf("Sum_f6.dbf") # Green land and square functional areas


colnames(Sumf1)[c(1:2)] <- c("Id", "cnt_f1")
colnames(Sumf2)[c(1:2)] <- c("Id", "cnt_f2")
colnames(Sumf3)[c(1:2)] <- c("Id", "cnt_f3")
colnames(Sumf4)[c(1:2)] <- c("Id", "cnt_f4")
colnames(Sumf5)[c(1:2)] <- c("Id", "cnt_f5")
colnames(Sumf6)[c(1:2)] <- c("Id", "cnt_f6")

head(Sumf1)
head(road_20)

road_20 <- data.frame(road_20)
road_20 <- road_20[, 1:2]
colnames(road_20)[1:2] <- c("Id", "area")

road_20 <- merge(x = road_20, y = Sumf1, by = "Id", all = TRUE)
road_20 <- merge(x = road_20, y = Sumf2, by = "Id", all = TRUE)
road_20 <- merge(x = road_20, y = Sumf3, by = "Id", all = TRUE)
road_20 <- merge(x = road_20, y = Sumf4, by = "Id", all = TRUE)
road_20 <- merge(x = road_20, y = Sumf5, by = "Id", all = TRUE)
road_20 <- merge(x = road_20, y = Sumf6, by = "Id", all = TRUE)
head(road_20)

road_20[is.na(road_20)] <- 0

# Calculate frequency density after adding weights
road_20$fi_f1 <- road_20$cnt_f1 / sum(road_20$cnt_f1) * 50
road_20$fi_f2 <- road_20$cnt_f2 / sum(road_20$cnt_f2) * 65
road_20$fi_f3 <- road_20$cnt_f3 / sum(road_20$cnt_f3) * 30
road_20$fi_f4 <- road_20$cnt_f4 / sum(road_20$cnt_f4) * 50
road_20$fi_f5 <- road_20$cnt_f5 / sum(road_20$cnt_f5) * 35
road_20$fi_f6 <- road_20$cnt_f6 / sum(road_20$cnt_f6) * 70
road_20$sum <- road_20$fi_f1 + road_20$fi_f2 + road_20$fi_f3 + road_20$fi_f4 + road_20$fi_f5 + road_20$fi_f6
head(road_20)


# Calculation type ratio
road_20$ci_f1 <- road_20$fi_f1 / road_20$sum
road_20$ci_f2 <- road_20$fi_f2 / road_20$sum
road_20$ci_f3 <- road_20$fi_f3 / road_20$sum
road_20$ci_f4 <- road_20$fi_f4 / road_20$sum
road_20$ci_f5 <- road_20$fi_f5 / road_20$sum
road_20$ci_f6 <- road_20$fi_f6 / road_20$sum

road_20$result <- ifelse(road_20$ci_f1 > 0.5, "f1",
  ifelse(road_20$ci_f2 > 0.5, "f2",
    ifelse(road_20$ci_f3 > 0.5, "f3",
      ifelse(road_20$ci_f4 > 0.5, "f4",
        ifelse(road_20$ci_f5 > 0.5, "f5",
          ifelse(road_20$ci_f6 > 0.5, "f6",
            ifelse(road_20$sum == 0, "NULL", "mix")
          )
        )
      )
    )
  )
)


head(road_20)
table(road_20$result)
functional <- road_20[, c("Id", "area", "result")]
head(road_20)

write.xlsx(functional, "data/UFS_classification_prediction/Ft20c.xlsx")


# Group by the result column and calculate the total area for each category.
result_totals <- road_20 %>%
  group_by(result) %>%
  summarise(area = sum(area))
result_totals

# Subdivide the mix function area and extract the mix function area data
road_mix <- road_20[, c("Id", "ci_f1", "ci_f2", "ci_f3", "ci_f4", "ci_f5", "ci_f6", "sum", "result")]
road_mix <- road_20[road_20$result == "mix", ]
road_mix <- road_mix[, c("Id", "ci_f1", "ci_f2", "ci_f3", "ci_f4", "ci_f5", "ci_f6", "sum", "result")]

# Check which rows have no missing values
complete_rows <- complete.cases(road_mix)

# Use this information to select a subset of the dataframe
road_mix <- road_mix[complete_rows, ]
head(road_mix)

colnames(road_mix)[2:7] <- c("f1", "f2", "f3", "f4", "f5", "f6")

# Extract the column name corresponding to the maximum value in each row from columns 2 to 7.
max <- apply(road_mix[, 2:7], 1, function(row) {
  max_value <- max(row)
  colnames(road_mix[, 2:7])[row == max_value]
})

# Add the results to the end of the data frame
road_mix <- cbind(road_mix, max)
head(road_mix)

# Extract the column name corresponding to the second largest value in each row of columns 2 to 7
second <- apply(road_mix[, 2:7], 1, function(row) {
  sorted_values <- sort(row, decreasing = TRUE)
  second <- sorted_values[2]
  colnames(road_mix[, 2:7])[row == second]
})

# Add the results to the end of the data frame
road_mix <- cbind(road_mix, second)

head(road_mix)
# Merge the results of two columns to create a hybrid functional area.
road_mix$combined <- paste(road_mix$max, road_mix$second, sep = "-") 
head(road_mix)

# Count the occurrences of each category
table(road_mix$combined)
road_mix2 <- road_mix[, c("Id", "combined")]

# Summarize the calculation results of urban functional zones
functional <- merge(x = functional, y = road_mix2, by = "Id", all = TRUE)

# Use the gsub function to replace strings
functional$combined <- gsub("f2-f1", "f1-f2", functional$combined)
functional$combined <- gsub("f3-f1", "f1-f3", functional$combined)
functional$combined <- gsub("f3-f2", "f2-f3", functional$combined)
functional$combined <- gsub("f4-f1", "f1-f4", functional$combined)
functional$combined <- gsub("f4-f2", "f2-f4", functional$combined)
functional$combined <- gsub("f4-f3", "f3-f4", functional$combined)
functional$combined <- gsub("f5-f1", "f1-f5", functional$combined)
functional$combined <- gsub("f5-f2", "f2-f5", functional$combined)
functional$combined <- gsub("f5-f3", "f3-f5", functional$combined)
functional$combined <- gsub("f5-f4", "f4-f5", functional$combined)
functional$combined <- gsub("f6-f1", "f1-f6", functional$combined)
functional$combined <- gsub("f6-f2", "f2-f6", functional$combined)
functional$combined <- gsub("f6-f3", "f3-f6", functional$combined)
functional$combined <- gsub("f6-f4", "f4-f6", functional$combined)
functional$combined <- gsub("f6-f5", "f5-f6", functional$combined)

table(functional$combined) # Count the occurrences of each category

# Calculate the area of each mixed-use zone
result_mix <- functional %>%
  group_by(combined) %>%
  summarise(area = sum(area))
result_mix


# Output Excel results
write.xlsx(functional, "data/UFS_classification_prediction/Ft20_mix5.xlsx")
write.xlsx(result_mix, "data/UFS_classification_prediction/reslut_mix5.xlsx")
write.xlsx(result_totals, "data/UFS_classification_prediction/reslut_totals5.xlsx")


# Reset rf.200

sum_ft <- read.dbf("data/UFS_classification_prediction/Sum_f8.dbf")
head(sum_ft)

# Create a function to generate result text based on conditions
compare_columns <- function(sum_ft) {
  result <- character(nrow(sum_ft))
  for (i in 1:nrow(sum_ft)) {
    if (is.na(sum_ft$First_Ft20[i]) && is.na(sum_ft$Last_Ft20_[i])) {
      result[i] <- NA
    } else if (is.na(sum_ft$First_Ft20[i]) && !is.na(sum_ft$Last_Ft20_[i])) {
      result[i] <- as.character(sum_ft$Last_Ft20_[i])
    } else if (!is.na(sum_ft$First_Ft20[i]) && is.na(sum_ft$Last_Ft20_[i])) {
      result[i] <- as.character(sum_ft$First_Ft20[i])
    } else if (!is.na(sum_ft$First_Ft20[i]) && !is.na(sum_ft$Last_Ft20_[i]) && sum_ft$First_Ft20[i] == sum_ft$Last_Ft20_[i]) {
      result[i] <- as.character(sum_ft$First_Ft20[i])
    } else {
      result[i] <- "mix"
    }
  }
  return(result)
}

# Save the results in the last column of Sum_ft
sum_ft$Comparison_Result <- compare_columns(sum_ft)
colnames(sum_ft)[1] <- "Id"
colnames(sum_ft)[5] <- "Functional"
rf.0324 <- sum_ft[, c(1, 5)]
head(rf.0324)

# Load factor data for rf.200

rf.200 <- read.xlsx("data/UFS_classification_prediction/rf.200.xlsx")

rf.200 <- rf.200[, -2]
rf.0324 <- merge(x = rf.0324, y = rf.200, by = "Id", all = TRUE)

tmp_20 <- read_sf("data/UFS_classification_prediction/tmp_2020.gpkg")

head(tmp_20)
tmp_20 <- data.frame(tmp_20)
tmp_20 <- tmp_20[, c(1, 5)]
colnames(tmp_20)[2] <- "tmp"
rf.0324 <- merge(x = rf.0324, y = tmp_20, by = "Id", all = TRUE)
head(rf.0324)
summary(rf.0324)
rf.0324 <- rf.0324[, -1]
rf.0324 <- rf.0324[-which(rf.0324$Functional == "mix"), ] # Remove the mixed zone
rf.0324 <- rf.0324[-which(rf.0324$Functional == "f6"), ] # Remove Greenland F6
summary(rf.0324)
rf.0324 <- na.omit(rf.0324) # Remove missing values
head(rf.0324)
table(rf.0324$Functional)


# 2. Urban functional zone modeling -----------------------------------------------------------------

# Building a Random Forest Model
# Divide the dataset into training and test sets
set.seed(666)
trainlist <- caret::createDataPartition(rf.0324$Functional, p = 0.8, list = FALSE) # Specify package, p represents the percentage of the whole.

# No need to specify the specific row number
trainset <- rf.0324[trainlist, ]
testset <- rf.0324[-trainlist, ]
dim(trainset)
dim(testset)

# Set seed value to avoid duplication; the selection of variables and samples is random, so that future models can be used.
# Set seed number
set.seed(666)


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
# The types that need to be predicted, so use class
rf.test <- predict(rf.train, newdata = testset, type = "class") 
rf.test

rf.cf <- caret::confusionMatrix(as.factor(rf.test), as.factor(testset$Functional))
rf.cf


# 3.ROC,AUC
rf.test2 <- predict(rf.train, newdata = testset, type = "prob")
# You can't directly use class to draw an ROC curve. Prob is the probability, which is very important; the probability distribution is 0-1 (the predicted probability of each flower in each class).

head(rf.test2)

roc.rf <- multiclass.roc(testset$Functional, rf.test2)
# Don't use ROC directly, this is a multi-class classification problem, don't use Plot directly

# Because this is a multi-class problem, you need to select one class to calculate
head(roc.rf)


# 3. Urban Functional Zone Simulation 2000-2020 -------------------------------------------------------

## 3.1 2020 Urban Land Use Forecast --------------------------------------------------------

dem <- read_sf("data/UFS_classification_prediction/2020/z_dem.gpkg")
gdp <- read_sf("data/UFS_classification_prediction/2020/z_gdp.gpkg")
ndvi <- read_sf("data/UFS_classification_prediction/2020/z_ndvi.gpkg")
dn <- read_sf("data/UFS_classification_prediction/2020/z_dn.gpkg")
pop <- read_sf("data/UFS_classification_prediction/2020/z_pop.gpkg")
slop <- read_sf("data/UFS_classification_prediction/2020/z_slop.gpkg")
tmp <- read_sf("data/UFS_classification_prediction/2020/z_t_2020.gpkg")

head(dn)

factor20 <- data.frame(
  dn$ID, dem$`_mean`, dn$`_mean`, ndvi$`_mean`, gdp$`_mean`, pop$`_mean`,
  slop$`_mean`
)
factor20 <- data.frame(factor20, tmp$`_mean`)
head(factor20)

colnames(factor20)[1:8] <- c("Id", "dem", "dn", "ndvi", "gdp", "pop", "slop", "tmp")
head(factor20)

colnames(factor20)[1:8] <- c("Id", "dem", "dn", "ndvi", "gdp", "pop", "slop", "tmp")
head(factor20)
factor20a <- factor20
factor20a <- na.omit(factor20a) # Remove missing values
head(factor20a)

# Replace ID with Functional to ensure the data format of the factor is consistent with the training set.
colnames(factor20a)[1] <- "Functional"
factor20a$Functional <- factor(factor20a$Functional)
head(factor20a)

set.seed(666)
rf.20 <- predict(rf.train, factor20a, type = "class")
head(rf.20)

data <- data.frame(rf.20)
data <- data.frame(factor20a, data)

data <- data[, c(1, 9)]
head(data)
colnames(data)[1] <- "Id"

factor20 <- merge(x = factor20, y = data, by = "Id", all = TRUE)
head(factor20)
table(factor20$rf.20)

write.xlsx(factor20, "data/UFS_classification_prediction/2020_Functional_Zone_predict.xlsx")


## 3.2 2015 Urban Land Use Forecast ---------------------------------------------------------
dem <- read_sf("2015/z_dem.gpkg")
gdp <- read_sf("2015/z_gdp.gpkg")
ndvi <- read_sf("2015/z_ndvi.gpkg")
dn <- read_sf("2015/z_dn.gpkg")
pop <- read_sf("2015/z_pop.gpkg")
slop <- read_sf("2015/z_slop.gpkg")
tmp <- read_sf("2015/z_t_2015.gpkg")

head(dn)

factor15 <- data.frame(
  dn$ID, dem$`_mean`, dn$`_mean`, ndvi$`_mean`, gdp$`_mean`, pop$`_mean`,
  slop$`_mean`
)
factor15 <- data.frame(factor15, tmp$`_mean`)
head(factor15)

colnames(factor15)[1:8] <- c("Id", "dem", "dn", "ndvi", "gdp", "pop", "slop", "tmp")
head(factor15)
factor15a <- factor15
factor15a <- na.omit(factor15a) # Remove missing values
head(factor15a)

# Replace ID with Functional to ensure the data format of the factor is consistent with the training set.
colnames(factor15a)[1] <- "Functional"
factor15a$Functional <- factor(factor15a$Functional)
head(factor15a)

set.seed(666)
rf.15 <- predict(rf.train, factor15a, type = "class")
head(rf.15)

data <- data.frame(rf.15)
data <- data.frame(factor15a, data)

data <- data[, c(1, 9)]
head(data)
colnames(data)[1] <- "Id"

factor15 <- merge(x = factor15, y = data, by = "Id", all = TRUE)
head(factor15)
table(factor15$rf.15)

write.xlsx(factor15, "data/UFS_classification_prediction/2015_Functional_Zone_predict.xlsx")

## 3.3 2010 Urban Land Use Forecast ---------------------------------------------------------

dem <- read_sf("data/UFS_classification_prediction/2010/z_dem.gpkg")
gdp <- read_sf("data/UFS_classification_prediction/2010/z_gdp.gpkg")
ndvi <- read_sf("data/UFS_classification_prediction/2010/z_ndvi.gpkg")
dn <- read_sf("data/UFS_classification_prediction/2010/z_dn.gpkg")
pop <- read_sf("data/UFS_classification_prediction/2010/z_pop.gpkg")
slop <- read_sf("data/UFS_classification_prediction/2010/z_slop.gpkg")
tmp <- read_sf("data/UFS_classification_prediction/2010/z_t_2010.gpkg")


head(dn)

factor10 <- data.frame(
  dn$ID, dem$`_mean`, dn$`_mean`, ndvi$`_mean`, gdp$`_mean`, pop$`_mean`,
  slop$`_mean`
)
factor10 <- data.frame(factor10, tmp$`_mean`)

head(factor10)

colnames(factor10)[1:8] <- c("Id", "dem", "dn", "ndvi", "gdp", "pop", "slop", "tmp")
head(factor10)
factor10a <- factor10
factor10a <- na.omit(factor10a) # Remove missing values

# Replace ID with Functional to ensure the data format of the factor is consistent with the training set.
colnames(factor10a)[1] <- "Functional"
factor10a$Functional <- factor(factor10a$Functional)
head(factor10a)

set.seed(300)
rf.10 <- predict(rf.train, factor10a, type = "class")
head(rf.10)

data <- data.frame(rf.10)
data <- data.frame(factor10a, data)

data <- data[, c(1, 9)]
head(data)
colnames(data)[1] <- "Id"

factor10 <- merge(x = factor10, y = data, by = "Id", all = TRUE)
head(factor10)
table(factor10$rf.10)

write.xlsx(factor10, "data/UFS_classification_prediction/2010_Functional_Zone_predict.xlsx")


## 3.4 2005 Urban Land Use Forecast ---------------------------------------------------------

dem <- read_sf("data/UFS_classification_prediction/2005/z_dem.gpkg")
gdp <- read_sf("data/UFS_classification_prediction/2005/z_gdp.gpkg")
ndvi <- read_sf("data/UFS_classification_prediction/2005/z_ndvi.gpkg")
dn <- read_sf("data/UFS_classification_prediction/2005/z_dn.gpkg")
pop <- read_sf("data/UFS_classification_prediction/2005/z_pop.gpkg")
slop <- read_sf("data/UFS_classification_prediction/2005/z_slop.gpkg")
tmp <- read_sf("data/UFS_classification_prediction/2005/z_t_2005.gpkg")

head(dn)

factor05 <- data.frame(
  dn$ID, dem$`_mean`, dn$`_mean`, ndvi$`_mean`, gdp$`_mean`, pop$`_mean`,
  slop$`_mean`
)
factor05 <- data.frame(factor05, tmp$`_mean`)

head(factor05)

colnames(factor05)[1:8] <- c("Id", "dem", "dn", "ndvi", "gdp", "pop", "slop", "tmp")
head(factor05)
factor05a <- factor05
factor05a <- na.omit(factor05a) # Remove missing values
head(factor05a)

# Replace ID with Functional to ensure the data format of the factor is consistent with the training set.
colnames(factor05a)[1] <- "Functional"
factor05a$Functional <- factor(factor05a$Functional)
head(factor05a)

set.seed(666)
rf.05 <- predict(rf.train, factor05a, type = "class")
head(rf.05)

data <- data.frame(rf.05)
data <- data.frame(factor05a, data)

data <- data[, c(1, 9)]
head(data)
colnames(data)[1] <- "Id"

factor05 <- merge(x = factor05, y = data, by = "Id", all = TRUE)
head(factor05)
table(factor05$rf.05)

write.xlsx(factor05, "data/UFS_classification_prediction/2005_Functional_Zone_predict.xlsx")


## 3.5 Result Prediction of Urban Land Use in 2000 ---------------------------------------------------------

dem <- read_sf("data/UFS_classification_prediction/2000/z_dem.gpkg")
gdp <- read_sf("data/UFS_classification_prediction/2000/z_gdp.gpkg")
ndvi <- read_sf("data/UFS_classification_prediction/2000/z_ndvi.gpkg")
dn <- read_sf("data/UFS_classification_prediction/2000/z_dn.gpkg")
pop <- read_sf("data/UFS_classification_prediction/2000/z_pop.gpkg")
slop <- read_sf("data/UFS_classification_prediction/2000/z_slop.gpkg")
tmp <- read_sf("data/UFS_classification_prediction/2000/z_t_2000.gpkg")

head(dn)

factor00 <- data.frame(
  dn$ID, dem$`_mean`, dn$`_mean`, ndvi$`_mean`, gdp$`_mean`, pop$`_mean`,
  slop$`_mean`
)
factor00 <- data.frame(factor00, tmp$`_mean`)

head(factor00)

colnames(factor00)[1:8] <- c("Id", "dem", "dn", "ndvi", "gdp", "pop", "slop", "tmp")
head(factor00)
factor00a <- factor00
factor00a <- na.omit(factor00a) # Remove missing values
head(factor00a)

# Replace ID with Functional to ensure the data format of the factor is consistent with the training set.
colnames(factor00a)[1] <- "Functional"
factor00a$Functional <- factor(factor00a$Functional)
head(factor00a)

set.seed(666)
rf.00 <- predict(rf.train, factor00a, type = "class")
head(rf.00)

data <- data.frame(rf.00)
data <- data.frame(factor00a, data)

data <- data[, c(1, 9)]
head(data)
colnames(data)[1] <- "Id"

factor00 <- merge(x = factor00, y = data, by = "Id", all = TRUE)
head(factor00)
table(factor00$rf.00)

write.xlsx(factor00, "data/UFS_classification_prediction/2000_Functional_Zone_predict.xlsx")
