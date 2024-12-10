#############################################################
# EdX HarvardX: PH125.9x Capstone: Choose Your Own (CYO) Project
# Create London crime Model using lsoa data set. 
############################################################

#Check & Install necessary packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(chron)) install.packages("chron", repos = "https://cran.rstudio.com/")

suppressWarnings({
  if (!requireNamespace("ROCR", quietly = TRUE)) {
    install.packages("ROCR")
  }
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    install.packages("xgboost")
  }
  if (!requireNamespace("tinytex", quietly = TRUE)) {
    install.packages("tinytex")
  }
  if (!tinytex::is_tinytex()) {
    tinytex::install_tinytex()
  }
})

#Installation of TinyTex is required to generate and format PDF output from rmarkdown (rmd) file.


# Load necessary libraries
library(dplyr)          # Data manipulation
library(ggplot2)        # Data visualization
library(caret)          # Machine learning
library(randomForest)   # Random forest model
library(chron)          # Date handling
library(ROCR)           # Performance evaluation
library(tidyr)          # Data tidying
library(xgboost)        # Gradient boosting model
library(tidyr)          # Data tidying
library(tidyverse)
library(Matrix)

# read data from local machine
data <- read.csv('C://Users//prero//Downloads//london_crime_by_lsoa.csv',header=TRUE)

# read data from github repository data folder
#data <- read.csv('./data/london_crime_by_lsoa.csv', header=TRUE)

#Subsetting the data to have 100k records only

data <- data[1:100000,]     #100k
#data <- data[1:1000000,]   #1M


# Data Cleaning

# Convert year and month into a Date format for easier manipulation
data$Date <- as.Date(paste(data$year, data$month, "01", sep = "-"), format="%Y-%m-%d")

# Remove rows with NA values in critical columns
data_clean <- data %>%
  filter(!is.na(value) & !is.na(year) & !is.na(month))

# Check data types and convert if necessary
data_clean$borough <- as.factor(data_clean$borough)
data_clean$major_category <- as.factor(data_clean$major_category)
data_clean$minor_category <- as.factor(data_clean$minor_category)

# Exploratory Data Analysis (EDA)

# Summary statistics
summary(data_clean)

# Distribution of values by major category
ggplot(data_clean, aes(x = major_category, y = value, fill = major_category)) +
  geom_boxplot() +
  labs(title = "Value Distribution by Major Category", x = "Major Category", y = "Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Time series plot of values over time
data_clean %>%
  group_by(Date) %>%
  summarise(mean_value = mean(value)) %>%
  ggplot(aes(x = Date, y = mean_value)) +
  geom_line() +
  labs(title = "Mean Value Over Time", x = "Date", y = "Mean Value")

# Distribution of values by borough
ggplot(data_clean, aes(x = borough, y = value, fill = borough)) +
  geom_boxplot() +
  labs(title = "Value Distribution by Borough", x = "Borough", y = "Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Distribution of values by minor category
ggplot(data_clean, aes(x = minor_category, y = value, fill = minor_category)) +
  geom_boxplot() +
  labs(title = "Value Distribution by Minor Category", x = "Minor Category", y = "Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Machine Learning Models

# Prepare data for modeling
set.seed(123)  # For reproducibility

# Create a train/test split
trainIndex <- createDataPartition(data_clean$value, p = .8, 
                                  list = FALSE, 
                                  times = 1)
data_train <- data_clean[trainIndex, ]
data_test  <- data_clean[-trainIndex, ]

# Convert categorical variables to numeric
data_train_matrix <- model.matrix(value ~ borough + major_category + minor_category + year + month - 1, data = data_train)
data_test_matrix <- model.matrix(value ~ borough + major_category + minor_category + year + month - 1, data = data_test)

# Convert to sparse matrices
sparse_train_matrix <- Matrix(data_train_matrix, sparse = TRUE)
sparse_test_matrix <- Matrix(data_test_matrix, sparse = TRUE)

# Prepare data for xgboost
dtrain <- xgb.DMatrix(data = sparse_train_matrix, label = data_train$value)
dtest <- xgb.DMatrix(data = sparse_test_matrix, label = data_test$value)

# Model 1: Linear Regression
linear_model <- lm(value ~ borough + major_category + minor_category + year + month, data = data_train)
summary(linear_model)

# Predict on test data
predictions_linear <- predict(linear_model, newdata = data_test)

# Model 2: Gradient Boosting Machine (XGBoost)
params <- list(
  objective = "reg:squarederror", # Regression task
  eval_metric = "rmse",           # Root Mean Squared Error
  max_depth = 6,                  # Depth of trees
  eta = 0.1,                      # Learning rate
  nthread = 2                     # Number of threads
)

xgb_model <- xgb.train(params = params,
                       data = dtrain,
                       nrounds = 100,         # Number of boosting rounds
                       verbose = 0)

# Predict on test data
predictions_xgb <- predict(xgb_model, dtest)

# Model 3: Random Forest
rf_model <- randomForest(value ~ borough + major_category + minor_category + year + month, data = data_train)

# Predict on test data for Random Forest
predictions_rf <- predict(rf_model, newdata = data_test)

# Evaluate the models

# For Linear Regression
mse_linear <- mean((predictions_linear - data_test$value)^2)
r2_linear <- 1 - (sum((predictions_linear - data_test$value)^2) / sum((data_test$value - mean(data_test$value))^2))
cat("Mean Squared Error for Linear Regression:", mse_linear, "\n")
cat("R-squared for Linear Regression:", r2_linear, "\n")

# For XGBoost
mse_xgb <- mean((predictions_xgb - data_test$value)^2)
r2_xgb <- 1 - (sum((predictions_xgb - data_test$value)^2) / sum((data_test$value - mean(data_test$value))^2))
cat("Mean Squared Error for XGBoost:", mse_xgb, "\n")
cat("R-squared for XGBoost:", r2_xgb, "\n")

# For Random Forest
mse_rf <- mean((predictions_rf - data_test$value)^2)
r2_rf <- 1 - (sum((predictions_rf - data_test$value)^2) / sum((data_test$value - mean(data_test$value))^2))
cat("Mean Squared Error for Random Forest:", mse_rf, "\n")
cat("R-squared for Random Forest:", r2_rf, "\n")

# Performance Comparison
cat("Model performance comparison:\n")
cat("Linear Regression MSE:", mse_linear, "\n")
cat("Linear Regression R-squared:", r2_linear, "\n")
cat("XGBoost MSE:", mse_xgb, "\n")
cat("XGBoost R-squared:", r2_xgb, "\n")
cat("Random Forest MSE:", mse_rf, "\n")
cat("Random Forest R-squared:", r2_rf, "\n")

###################################################################################