# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\anton\OneDrive\Documents\GitHub\MachineLearningP7\Assignment_2\2.2\US-pumpkins.csv")  # Replace with the actual path

# Define the features
features = ['City Name', 'Type', 'Package', 'Variety', 'Sub Variety', 'Grade', 'Low Price', 'High Price', 'Mostly Low', 'Mostly High', 'Item Size', 'Color', 'Unit of Sale', 'Quality', 'Condition', 'Appearance', 'Storage', 'Crop', 'Repack']

# Define the target variable
target = 'Trans Mode'

# Extract the relevant features and target variable
data = data[features + [target]]

# Data Preprocessing
# Handle missing values for numerical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# Handle missing categorical values, for example, by filling with a placeholder
data['Type'].fillna('Unknown', inplace=True)

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
for column in ['City Name', 'Type', 'Package', 'Variety', 'Sub Variety', 'Grade', 'Item Size', 'Color', 'Unit of Sale', 'Quality', 'Condition', 'Appearance', 'Storage', 'Crop', 'Repack']:
    data[column] = label_encoder.fit_transform(data[column])

# Convert the 'Trans Mode' column to integers
data[target] = label_encoder.fit_transform(data[target])

# Separate the target variable
X = data.drop(target, axis=1)
y = data[target]

# Split the data into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate and visualize the ROC curve
y_scores = model.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()