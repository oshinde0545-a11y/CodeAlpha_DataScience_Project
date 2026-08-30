from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

print("Iris dataset loaded successfully!")
print("Features:", iris.feature_names)
print("Target names:", iris.target_names)
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Input features
X = iris.data

# Target/output
y = iris.target

print("Input data shape:", X.shape)
print("Target data shape:", y.shape)

print("First flower measurements:", X[0])
print("First flower target:", y[0])

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()

# Input and output
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()

# Input and output
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred)

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# New flower measurements
new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict species
prediction = model.predict(new_flower)

# Convert number to species name
species = iris.target_names[prediction[0]]

print("Predicted Species:", species)