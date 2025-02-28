import pandas as pd

# Load the dataset
df = pd.read_excel('AmesHousing.xlsx')

# Display the first few rows
print(df.head())

# Show summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Drop columns with excessive missing values (if necessary)
df.dropna(axis=1, thresh=len(df) * 0.8, inplace=True)

# Encode categorical features
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Define features and target variable
X = df.drop(columns=['SalePrice'])  # Drop the target variable
y = df['SalePrice']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Root Mean Squared Error: {rmse}')

# Save the model for deployment
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
