# MLE vs MAP Logistic Regression

This project compares **Maximum Likelihood Estimation (MLE)** and **Maximum A Posteriori (MAP)** using Logistic Regression.

The **Breast Cancer Wisconsin dataset** from Scikit-learn is used for classification.

## Models Used

* **MLE** – Logistic Regression without regularization
* **MAP (L1)** – Logistic Regression with L1 regularization
* **MAP (L2)** – Logistic Regression with L2 regularization

## Dataset

The dataset contains:

* 569 samples
* 30 features
* 2 classes

## Steps

1. Load the breast cancer dataset
2. Split the data into training and testing sets
3. Standardize the features
4. Train MLE, MAP L1, and MAP L2 models
5. Calculate model accuracy
6. Compare the learned coefficients

## Requirements

Install the required libraries:

```bash
pip install pandas scikit-learn
```

## Run

```bash
python main.py
```

## Output

The program displays:

* Accuracy of MLE
* Accuracy of MAP (L1)
* Accuracy of MAP (L2)
* First 10 feature coefficients
* Number of zero coefficients for L1 and L2

## Conclusion

This project shows how **regularization affects Logistic Regression coefficients and model performance**.

L1 regularization can make some coefficients zero, while L2 regularization mainly reduces the coefficient values.
