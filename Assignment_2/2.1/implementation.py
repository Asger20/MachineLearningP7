import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_linnerud

# Load the Linnerud dataset
data = load_linnerud()
X = data.data  # Features (including Waist)
y = data.target[:, 0]  # Target variable (Situps)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Plot the actual vs. predicted situps
plt.scatter(X_test[:, 0], y_test, label="Actual", color='blue')
plt.scatter(X_test[:, 0], y_pred, label="Predicted", color='red')
plt.xlabel("Waistline")
plt.ylabel("Situps")
plt.legend()
plt.title(f"Actual vs. Predicted Situps (MSE: {mse:.2f})")
plt.show()

from sklearn.metrics import mean_squared_error

# Assuming you have trained your linear regression model (as shown in the previous code example)
# And you have made predictions on the test data

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")