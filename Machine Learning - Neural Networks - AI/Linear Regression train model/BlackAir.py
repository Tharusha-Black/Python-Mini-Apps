import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_curve, auc, accuracy_score

# Load the dataset
blackair_df = pd.read_csv("BlackAir.csv")
metadata_df = pd.read_csv("Metadata.csv")

# Display first few rows of the dataset
print("BlackAir Dataset:")
print(blackair_df.head())

# Display metadata
print("\nMetadata:")
print(metadata_df)

# Check for missing data in the BlackAir dataset
missing_data_count = blackair_df.isnull().sum()

# Print the count of missing values for each column
print("\nMissing Data in BlackAir Dataset:")
print(missing_data_count)


# Distribution of satisfied customers by age
sns.histplot(data=blackair_df, x='Age', hue='Satisfaction', multiple='stack', bins=20)
plt.title('Distribution of Satisfied Customers by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Distribution of satisfied customers by gender
sns.countplot(data=blackair_df, x='Gender', hue='Satisfaction')
plt.title('Distribution of Satisfied Customers by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Compare satisfaction levels between business and personal travel
sns.countplot(data=blackair_df, x='Travel type', hue='Satisfaction')
plt.title('Satisfaction Levels by Travel Type')
plt.xlabel('Travel Type')
plt.ylabel('Count')
plt.show()

# Convert 'Satisfaction' column to numeric values
blackair_df['Satisfaction'] = blackair_df['Satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

# Correlation between flight details and satisfaction
flight_details = ['Flight distance', 'Departure delay in minutes', 'Arrival delay in minutes']
flight_corr = blackair_df[flight_details + ['Satisfaction']].corr()

# Visualize correlation matrix
sns.heatmap(flight_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Flight Details and Satisfaction')
plt.show()

# Impact of in-flight services on satisfaction
inflight_services = ['Inflight wifi', 'Seat comfort', 'Food and drink']
for service in inflight_services:
    sns.boxplot(data=blackair_df, x='Satisfaction', y=service)
    plt.title(f'Impact of {service} on Satisfaction')
    plt.xlabel('Satisfaction')
    plt.ylabel(service)
    plt.show()

# Additional Insights

# 1. Customer Type and Satisfaction
sns.countplot(data=blackair_df, x='Customer type', hue='Satisfaction')
plt.title('Satisfaction Levels by Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Count')
plt.show()

# 2. Ticket Type and Satisfaction
sns.countplot(data=blackair_df, x='Ticket type', hue='Satisfaction')
plt.title('Satisfaction Levels by Ticket Type')
plt.xlabel('Ticket Type')
plt.ylabel('Count')
plt.show()

# 3. Booking Service and Satisfaction
sns.boxplot(data=blackair_df, x='Satisfaction', y='Booking service')
plt.title('Impact of Booking Service on Satisfaction')
plt.xlabel('Satisfaction')
plt.ylabel('Booking Service Rating')
plt.show()

# 4. Airport Check-in and Satisfaction
sns.boxplot(data=blackair_df, x='Satisfaction', y='Airport checkin')
plt.title('Impact of Airport Check-in on Satisfaction')
plt.xlabel('Satisfaction')
plt.ylabel('Airport Check-in Rating')
plt.show()

# 5. Cabin Crew Service and Satisfaction
sns.boxplot(data=blackair_df, x='Satisfaction', y='Cabin crew service')
plt.title('Impact of Cabin Crew Service on Satisfaction')
plt.xlabel('Satisfaction')
plt.ylabel('Cabin Crew Service Rating')
plt.show()

# Step 1: Load the dataset
blackair_df = pd.read_csv("BlackAir.csv")

# Step 2: Drop unnecessary columns for model training
blackair_df.drop(columns=["Module_id", "Response_ID"], inplace=True)

# Step 3: Convert categorical variables to numerical using one-hot encoding
blackair_df = pd.get_dummies(blackair_df, columns=["Gender", "Customer type", "Travel type", "Ticket type"])

# Step 4: Convert satisfaction column to binary (0 for dissatisfied, 1 for satisfied)
blackair_df["Satisfaction"] = blackair_df["Satisfaction"].map({"neutral or dissatisfied": 0, "satisfied": 1})

# Step 5: Handle missing values
imputer = SimpleImputer(strategy='mean')
blackair_df_imputed = pd.DataFrame(imputer.fit_transform(blackair_df), columns=blackair_df.columns)

# Print the dataset after handling missing values
print("Dataset after handling missing values:\n", blackair_df_imputed.head())

# Count missing values in the test set
missing_values = np.sum(blackair_df_imputed.isnull().sum())

# Print the count of missing values in the dataset
print("Number of missing values in the dataset:", missing_values)

# Step 6: Feature Selection (Select relevant features based on the analysis)
selected_features = ['Flight distance', 'Departure delay in minutes', 'Arrival delay in minutes',
                     'Flight schedule suitability', 'Booking service', 'Online checkin',
                     'Airport checkin', 'Baggage handling', 'Boarding service', 'Cabin crew service',
                     'Seat comfort', 'Food and drink', 'Inflight entertainment', 'Inflight wifi',
                     'Leg room', 'Inflight amenities quality', 'Cleanliness', 'Gender_Female', 'Gender_Male',
                     'Customer type_Discontinued', 'Customer type_Loyal', 'Travel type_Business',
                     'Travel type_Personal', 'Ticket type_Business', 'Ticket type_Economy', 'Ticket type_Economy Plus']

# Step 7: Split Data
X = blackair_df_imputed[selected_features]
y = blackair_df_imputed["Satisfaction"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Step 9: Evaluate Model
y_pred = linear_model.predict(X_test)
# Convert predicted probabilities to binary classifications
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print mean squared error and R-squared score
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
# Compute accuracy
accuracy = accuracy_score(y_test, y_pred_binary)

# Print accuracy
print("Accuracy:", accuracy)

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
blackair_df = pd.read_csv("BlackAir.csv")

# Step 2: Drop unnecessary columns for model training
blackair_df.drop(columns=["Module_id", "Response_ID"], inplace=True)

# Step 3: Convert categorical variables to numerical using one-hot encoding
blackair_df = pd.get_dummies(blackair_df, columns=["Gender", "Customer type", "Travel type", "Ticket type"])

# Step 4: Convert satisfaction column to binary (0 for dissatisfied, 1 for satisfied)
blackair_df["Satisfaction"] = blackair_df["Satisfaction"].map({"neutral or dissatisfied": 0, "satisfied": 1})

# Step 5: Handle missing values
imputer = SimpleImputer(strategy='mean')
blackair_df_imputed = pd.DataFrame(imputer.fit_transform(blackair_df), columns=blackair_df.columns)

# Step 6: Feature Selection (Select relevant features based on the analysis)
selected_features = ['Flight distance', 'Departure delay in minutes', 'Arrival delay in minutes',
                     'Flight schedule suitability', 'Booking service', 'Online checkin',
                     'Airport checkin', 'Baggage handling', 'Boarding service', 'Cabin crew service',
                     'Seat comfort', 'Food and drink', 'Inflight entertainment', 'Inflight wifi',
                     'Leg room', 'Inflight amenities quality', 'Cleanliness', 'Gender_Female', 'Gender_Male',
                     'Customer type_Discontinued', 'Customer type_Loyal', 'Travel type_Business',
                     'Travel type_Personal', 'Ticket type_Business', 'Ticket type_Economy', 'Ticket type_Economy Plus']

# Step 7: Split Data
X = blackair_df_imputed[selected_features]
y = blackair_df_imputed["Satisfaction"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Step 9: Evaluate Model
y_pred = linear_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Convert predicted probabilities to binary classifications
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Classification Report:\n", classification_report(y_test, y_pred_binary))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC Score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

# Feature Importance Plot (Not applicable for Linear Regression)
# Hyperparameter Tuning (Not applicable for Linear Regression)
# Cross-Validation
cv_scores = cross_val_score(linear_model, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Model Comparison (Not applicable for Linear Regression)
# Business Impact Analysis (Can be discussed based on model's predictions)

