# Hypertension Prevention Project

This project aims to predict the probability of a patient having high blood pressure using a machine learning model based on a dataset of patients.

## Code Description

The provided code is written in Python and utilizes the following libraries:

- `pandas`: For data manipulation.
- `scikit-learn`: For training and evaluating a random forest classification model.
- `numpy`: For mathematical operations.

The code workflow is as follows:

1. Load the dataset from a CSV file named `Valores_Hipertension.csv`.
2. Convert categorical variables to numerical ones.
3. Split the data into features (X) and labels (y).
4. Split the dataset into training and testing sets.
5. Train a `RandomForestClassifier` classification model with the best hyperparameters found.
6. Calculate the importance of each feature in prediction.
7. Make predictions on the test set and calculate the model's accuracy.

## Code Usage

To use this code, make sure you have the libraries mentioned in the `requirements.txt` file installed. Then, run the Python script `predict_hypertension.py`. The script will prompt you to enter information about a new patient and provide a prediction of whether the patient has high blood pressure or not.

## Example Data

In the `Valores_Hipertension.csv` file, you will find sample patient data, including information such as age, gender, BMI, blood pressure, family history, among others. The file also includes a column called `Hypertension`, indicating whether the patient has high blood pressure (Yes/No).

## Contribution

If you wish to contribute to this project, we encourage you to do so! You can open issues or send pull requests to propose improvements or corrections.
