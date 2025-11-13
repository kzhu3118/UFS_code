## Overview

This repository contains the R source code for the paper titled "Unveiling Long-term Dynamics of Urban Functional Spaces in Ultra-fast Growing Cities through Multisource Data Integration".

Contact: E-mail: luhw@igsnrr.ac.cn


## Data Availability

The dataset used in this study (approx. 41.2 MB) is archived on Zenodo due to file size limitations.

Data Link: https://zenodo.org/records/17595677

Instructions:
1.  Download the dataset (`data.zip`) from the Zenodo link above.
2.  Create a new folder named `data` in the root directory of this project.
3.  Unzip all files into the `/data` folder.

The scripts in this repository are configured to read data from this `/data` folder.

## Environment and Requirements

This code was tested in R (v4.5.2). Please ensure you have all the following R packages installed before running the scripts:

 sf
 sp
 raster
 foreign
 openxlsx
 dplyr
 caret
 pROC
 randomForest
 ggRandomForests
 kknn
 e1071
 skimr
 DataExplorer
 tidyverse
 kernlab
 xgboost
 Matrix
 rpart
 rpart.plot
 vivid

## How to Run

This repository contains two main R scripts. Please run them in the following order.

### Script 1: `UFS_classification_prediction.R`

Purpose: This is the main analysis pipeline for the paper. It performs data preprocessing, trains the final Random Forest model, and generates the Urban Functional Space (UFS) prediction maps for 2000-2020.

 Part 1: Data Preprocessing. Loads raw data, calculates functional densities and types, and identifies mixed functional zones.
 Part 2: Random Forest Model Training. Uses the processed data to train the final model used for prediction.
 Part 3: UFS Simulation (2000-2020). Loads the factor data for each 5-year interval, applies the trained model from Part 2, and outputs the final prediction results (e.g., `data/UFS_classification_prediction/2020_Functional_Zone_predict.xlsx`).

### Script 2: `Machine_learning_model.R`

Purpose: This is a  analysis script used to compare the performance of different machine learning models (as shown in Figure 5 and Figure 6 of the paper) to justify the selection of Random Forest (RF).

 Part 1-5: Builds and evaluates five different models: KNN, SVM, XGBoost, Random Forest, and Decision Tree.
 Note: The Random Forest model in Part 4 of this script is for a fair comparison against the other algorithms on the same test set (e.g., for generating ROC curves and confusion matrices). It is separate from the final prediction model in `UFS_classification_prediction.R`.
