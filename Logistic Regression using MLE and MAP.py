import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mle = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=1000
)

map_l1 = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    max_iter=1000
)

map_l2 = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    C=1.0,
    max_iter=1000
)

mle.fit(X_train, y_train)
map_l1.fit(X_train, y_train)
map_l2.fit(X_train, y_train)

pred_mle = mle.predict(X_test)
pred_l1 = map_l1.predict(X_test)
pred_l2 = map_l2.predict(X_test)

print("\nModel Accuracy")
print("-" * 30)
print(f"MLE      : {accuracy_score(y_test, pred_mle):.4f}")
print(f"MAP (L1) : {accuracy_score(y_test, pred_l1):.4f}")
print(f"MAP (L2) : {accuracy_score(y_test, pred_l2):.4f}")

coef_df = pd.DataFrame({
    "Feature": data.feature_names,
    "MLE": mle.coef_[0],
    "MAP (L1)": map_l1.coef_[0],
    "MAP (L2)": map_l2.coef_[0]
}).round(4)

print("\nLearned Coefficients (First 10 Features)")
print("-" * 75)
print(coef_df.head(10).to_string(index=False))

print("\nCoefficient Summary")
print("-" * 30)
print(f"Total Features           : {len(data.feature_names)}")
print(f"Zero Coefficients (L1)   : {(map_l1.coef_[0] == 0).sum()}")
print(f"Zero Coefficients (L2)   : {(map_l2.coef_[0] == 0).sum()}")