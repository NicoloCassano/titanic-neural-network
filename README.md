# Titanic Survival Prediction with Dense Neural Networks

## Objective
The objective of this project is to build and evaluate a **Dense Neural Network (DNN)**
to predict passenger survival on the Titanic dataset.
The task is framed as a **binary classification problem**.

## Dataset
The project uses the Titanic dataset, which contains demographic and travel information
about passengers aboard the Titanic.
The target variable is `Survived` (0 = Did not survive, 1 = Survived).

The dataset is publicly available on Kaggle.

## Pipeline
The project follows an end-to-end machine learning workflow:

- Handling missing values (`Age`, `Embarked`)
- Feature scaling using `StandardScaler`
- One-hot encoding of categorical variables
- Train/test split
- Dense Neural Network training using TensorFlow/Keras
- Model evaluation using accuracy and confusion matrix
- Visualization of training and validation metrics

## Model Architecture
- Input layer based on preprocessed features
- Two hidden Dense layers with ReLU activation
- Output layer with Sigmoid activation
- Loss function: Binary Crossentropy
- Optimizer: Adam

## Results
The model achieves good performance on both training and test sets.
Training and validation curves show a stable learning process with mild overfitting
after several epochs.

## Technologies
- Python
- TensorFlow / Keras
- scikit-learn
- pandas, numpy
- seaborn, matplotlib
  
