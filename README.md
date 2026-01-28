# Titanic Survival Prediction with Dense Neural Networks

## Project Overview
This project focuses on predicting passenger survival on the Titanic dataset using a Dense Neural Network (DNN).
The goal is to apply an end-to-end machine learning workflow, from data preprocessing to model training and evaluation, with attention to feature engineering and model generalization.

The task is framed as a binary classification problem.

## Dataset
The project uses the public Titanic dataset from Kaggle, which contains demographic and travel-related information about passengers aboard the Titanic.
The target variable is `Survived` (0 = Did not survive, 1 = Survived).

## Methodology
The workflow follows a structured machine learning pipeline:

- Handling missing values (Age, Embarked) using statistical imputation
- Feature scaling with StandardScaler to ensure stable neural network training
- One-hot encoding of categorical variables
- Train/Test split to evaluate model generalization
- Dense Neural Network training using TensorFlow/Keras

## Model Architecture

- Input layer based on preprocessed numerical and categorical features
- Two hidden Dense layers with ReLU activation
- Output layer with Sigmoid activation for binary classification
- Loss function: Binary Crossentropy
- Optimizer: Adam

 ## Results
The model shows solid performance on unseen data:

- **Train Accuracy:** 0.838  
- **Test Accuracy:** 0.827

## Visualizations
The project includes the following visual outputs:

- Training and validation loss/accuracy curves
- Confusion matrix for classification performance analysis

The confusion matrix highlights a balanced classification behavior, with a good ability to correctly
identify both surviving and non-surviving passengers.
Training and validation curves indicate a stable learning process, with only mild overfitting
observed after several epochs, suggesting reasonable generalization.

## Technologies
- Python
- TensorFlow / Keras
- scikit-learn
- pandas, numpy
- seaborn, matplotlib
  
