##  Data Mining Assignments (Classification, Clustering, MLP)

![](pic.png)

# Data Mining Project README

## Project Description
This project, focuses on analyzing a dataset related to blood donation. The dataset is named "transfusion.data" and contains information about individuals' blood donation history, including factors like recency, frequency, monetary donations, time since the last donation, and whether they donated blood in March 2007. 

---
# Assignment2 - Part1
## Project Overview
This project involves the analysis of maternal health risk data using various machine learning algorithms. The primary goal is to build predictive models for risk assessment based on the provided dataset named "Maternal Health Risk Data Set.csv."

## Project Components

### 1. Data Preprocessing
#### 1.1. Feature Selection
A subset of relevant features is selected and stored in a new DataFrame named `new_data_frame`. This subset includes columns 0 to 5 from the original dataset.

#### 1.2. Encoding Target Variable
The target variable, representing maternal health risk levels, is encoded into numerical values as follows:
- 'low risk' is encoded as 0
- 'mid risk' is encoded as 1
- 'high risk' is encoded as 2

### 2 Train-Test Split
The dataset is split into training and test sets. Approximately 70% of the data is used for training, and 30% is reserved for testing. Random indices are selected to achieve this split.

### 3. Model Building and Evaluation
The project employs several machine learning algorithms for modeling and evaluation. Each algorithm is explored with different parameter settings.
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
  
### 4. Model Evaluation Metrics
For each model, the following evaluation metrics are calculated:
- Accuracy on the test set
- Accuracy on the training set
- Cross-validation score (average accuracy)
- Confusion matrix
---
# Assignment2 - Part2

## Project Overview
This project involves the analysis of speech features data for the classification of speech disorders. The primary objective is to build and evaluate machine learning models to classify speech samples into two categories: healthy (1) and speech disorders (0). The dataset used for this project is named "pd_speech_features.csv." 

## Project Components

### 1. Data Preprocessing
#### 1.1. Feature Selection
A subset of relevant features is selected and stored in a new DataFrame named `new_data_frame`. This subset includes all columns except the first one, which is assumed to be an identifier.

#### 1.2. Train-Test Split
The dataset is divided into a training set and a test set. Approximately 70% of the data is used for training, and 30% is set aside for testing. Randomly selected rows are used for each split.

#### 1.3. Encoding Target Variable
The target variable, representing the classification labels, is encoded as follows:
- Healthy speech (Class 1) is encoded as 1.
- Speech disorders (Class 0) are encoded as 0.

### 2. Model Building and Evaluation
Machine learning models are built and evaluated using the training and test datasets. Three types of models are explored: Decision Trees, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forests. Different parameter settings are tested for each model.

### 3. Model Evaluation Metrics
For each model, the following evaluation metrics are calculated:
- Accuracy on the test set
- Accuracy on the training set
- Cross-validation score (average accuracy)
- Confusion matrix



