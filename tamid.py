import pandas as pd
import numpy as np
from sklearn import datasets

print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)
print("Scikit-learn version:", datasets.__doc__)

# Step 1: Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal and petal measurements)
y = iris.target  # Labels (setosa, versicolor, virginica)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train a classifier (e.g., K-Nearest Neighbors)
knn = KNeighborsClassifier(n_neighbors=3)  # Number of neighbors can be adjusted
knn.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = knn.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Weighted for multiclass classification
recall = recall_score(y_test, y_pred, average='weighted')

# Step 7: Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

