# Depression Identification Project

This project implements a robust machine learning pipeline to solve a binary classification problem inspired by the Kaggle Playground Series - Season 4, Episode 11. The final model achieved an impressive accuracy of 94.008% on the test set.

## Table of Contents
- [Project Inspiration](#project-inspiration)
- [Dataset](#dataset)
  - [Overview](#overview)
  - [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
  - [Architecture](#architecture)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Training and Evaluation](#training-and-evaluation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Key Results](#key-results)
- [Acknowledgments](#acknowledgments)

## Project Inspiration

This project was inspired by the Kaggle Playground Series - Season 4, Episode 11. The task involves building a binary classification model to predict an outcome based on provided features. The focus was on exploring feature engineering, hyperparameter optimization, and effective model design to achieve high accuracy.

## Dataset

### Overview

The dataset includes:
- 17 features representing various numerical and categorical variables.
- Binary target variable (0 or 1).

### Preprocessing
- **Handling Missing Values**: Imputed using statistical methods.
- **Categorical Encoding**: One-hot and ordinal encoding applied where necessary.
- **Scaling**: Numerical features were standardized to ensure consistent feature ranges.

## Feature Engineering

### Steps Taken:
- **Feature Selection**:
  - Identified and retained the most relevant features using correlation analysis and feature importance scores.
- **Feature Transformation**:
  - Log transformation and polynomial features were tested to improve model performance.
- **Interaction Features**:
  - Combined existing features to create new informative variables.

## Model Development

### Architecture
The final model uses a deep neural network implemented in TensorFlow and Keras:
- **Input Layer**: Batch normalization to normalize the 17 input features.
- **Hidden Layers**:
  - Four Dense layers with ReLU activation.
  - Dropout for regularization.
  - Batch normalization for stability.
- **Output Layer**: Single neuron with sigmoid activation for binary classification.

### Hyperparameter Tuning
- **Tuned Parameters**:
  - Number of neurons in hidden layers.
  - Dropout rates.
  - Learning rate.
- **Method**: Hyperband search using Keras Tuner.

## Training and Evaluation

### Training
- **Loss Function**: Binary Cross-Entropy.
- **Optimizer**: Adam.
- **Batch Size**: 32.
- **Epochs**: 20 with early stopping based on validation loss.

### Evaluation
Achieved a final test accuracy of **94.008%**.

Additional metrics computed:
- **Precision**: 0.91
- **Recall**: 0.92
- **F1-Score**: 0.915

### Results Visualization
- Loss and accuracy curves were plotted to ensure convergence and detect overfitting.
- ROC Curve and Precision-Recall Curve were analyzed to fine-tune the decision threshold.

## Key Results
- **Model Accuracy**: 94.008%
- **Interpretability**:
  - SHAP (SHapley Additive exPlanations) values were used to understand feature contributions.
- **Threshold Optimization**:
  - Decision threshold adjusted based on precision-recall tradeoff for optimal classification performance.

## Acknowledgments
Special thanks to the Kaggle community for the inspiration and dataset. The TensorFlow and Keras frameworks made it possible to implement and optimize the deep learning model efficiently.
