import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("train (1).csv")
print(df.columns)  # Check available columns

# Select relevant columns and clean
df = df[["OverallQual", "SalePrice"]].dropna()

# Split into features and target
X = df[["OverallQual"]].values
y = df["SalePrice"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train custom linear regression (least squares)
def train_custom_linear_regression(X, y):
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept

slope, intercept = train_custom_linear_regression(X_train, y_train)

def predict_custom(X, slope, intercept):
    return slope * X + intercept

# Custom model predictions and MSE
y_pred_custom_test = predict_custom(X_test, slope, intercept)
mse_custom = mean_squared_error(y_test, y_pred_custom_test)
print("Custom Model MSE:", mse_custom)

# Sklearn model training and testing
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_sklearn_test = model.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn_test)
print("Sklearn Model MSE:", mse_sklearn)

# Plot - Training set comparison
plt.scatter(X_train, y_train, label="Actual", alpha=0.6)
plt.plot(X_train, predict_custom(X_train, slope, intercept), color='red', label="Custom Model")
plt.plot(X_train, model.predict(X_train), color='green', linestyle='--', label="Sklearn Model")
plt.title("Train Set: Custom vs Sklearn Linear Regression")
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
plt.legend()
plt.show()

# Plot - Test set comparison
plt.scatter(X_test, y_test, label="Actual", alpha=0.6)
plt.plot(X_test, y_pred_custom_test, color='red', label="Custom Model")
plt.plot(X_test, y_pred_sklearn_test, color='green', linestyle='--', label="Sklearn Model")
plt.title("Test Set: Custom vs Sklearn Linear Regression")
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
plt.legend()
plt.show()

# QUESTION 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Load data
df = pd.read_csv("StudentsPerformance.csv")

# Preprocess: create binary target PassedMath
df["PassedMath"] = df["math score"].apply(lambda x: 1 if x >= 50 else 0)

# Use only reading score as feature
X = df[["reading score"]].values
y = df["PassedMath"].values

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Custom Logistic Regression -----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.01, iterations=1000):
    m, n = X.shape
    X_bias = np.hstack((np.ones((m, 1)), X))  # add bias term
    weights = np.zeros(n + 1)

    for _ in range(iterations):
        z = np.dot(X_bias, weights)
        predictions = sigmoid(z)
        error = predictions - y
        gradient = np.dot(X_bias.T, error) / m
        weights -= lr * gradient

    return weights

def predict(X, weights):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    probs = sigmoid(np.dot(X_bias, weights))
    return (probs >= 0.5).astype(int), probs

# Train custom model
custom_weights = train_logistic_regression(X_train, y_train)

# Predict on test
custom_preds, custom_probs = predict(X_test, custom_weights)

# Evaluate custom model
custom_acc = accuracy_score(y_test, custom_preds)
custom_prec = precision_score(y_test, custom_preds)
custom_rec = recall_score(y_test, custom_preds)
custom_f1 = f1_score(y_test, custom_preds)
print("Custom Logistic Regression:")
print(f"Accuracy: {custom_acc:.4f}, Precision: {custom_prec:.4f}, Recall: {custom_rec:.4f}, F1-Score: {custom_f1:.4f}")
ConfusionMatrixDisplay(confusion_matrix(y_test, custom_preds), display_labels=[0,1]).plot()
plt.title("Custom Model Confusion Matrix")
plt.show()

# ----- Sklearn Logistic Regression -----
sk_model = LogisticRegression()
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)

# Evaluate sklearn model
sk_acc = accuracy_score(y_test, sk_preds)
sk_prec = precision_score(y_test, sk_preds)
sk_rec = recall_score(y_test, sk_preds)
sk_f1 = f1_score(y_test, sk_preds)
print("Sklearn Logistic Regression:")
print(f"Accuracy: {sk_acc:.4f}, Precision: {sk_prec:.4f}, Recall: {sk_rec:.4f}, F1-Score: {sk_f1:.4f}")
ConfusionMatrixDisplay(confusion_matrix(y_test, sk_preds), display_labels=[0,1]).plot()
plt.title("Sklearn Model Confusion Matrix")
plt.show()



