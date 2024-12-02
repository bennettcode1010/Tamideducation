# Import necessary libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Load the digits dataset
digits = load_digits()
X = digits.data  # Features (flattened pixel values)
y = digits.target  # Labels (digit classes)

# Step 2: Preprocess the data by scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scales each feature to have mean=0 and std=1

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 4: Train a K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=3)  # Number of neighbors can be adjusted
knn.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = knn.predict(X_test)

# Step 6: Evaluate the model using confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred, target_names=digits.target_names.astype(str))

# Print results
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report_output)
