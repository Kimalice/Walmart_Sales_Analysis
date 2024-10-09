# Step 1: Setting Up the Environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 2: Load the Dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
features = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")

# Merge datasets on Store and Date
train_merged = pd.merge(train, features, on=["Store", "Date"], how="left")
train_merged = pd.merge(train_merged, stores, on="Store", how="left")

# Convert Date column to datetime format
train_merged['Date'] = pd.to_datetime(train_merged['Date'])

# Check basic information of the merged data
print(train_merged.info())

# Step 3: Exploratory Data Analysis (EDA)
# 3.1 Visualize Sales Trends Over Time
plt.figure(figsize=(15, 6))
sns.lineplot(data=train_merged, x="Date", y="Weekly_Sales", hue="Store", legend=None)
plt.title("Weekly Sales Trends by Store")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.show()

# 3.2 Analyze Sales During Holidays
train_merged['IsHoliday'] = train_merged['IsHoliday'].astype(int)
sns.boxplot(x='IsHoliday', y='Weekly_Sales', data=train_merged)
plt.title("Sales Distribution on Holidays vs. Non-Holidays")
plt.show()

# 3.3 Correlation Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(train_merged.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 3.4 Segment Analysis by Store Type
store_sales = train_merged.groupby('Type')['Weekly_Sales'].mean().reset_index()
sns.barplot(x='Type', y='Weekly_Sales', data=store_sales)
plt.title("Average Weekly Sales by Store Type")
plt.show()

# Step 4: Feature Engineering
# 4.1 Create Lag Features to Capture Seasonality
train_merged['Weekly_Sales_Lag_1'] = train_merged.groupby('Store')['Weekly_Sales'].shift(1)
train_merged['Weekly_Sales_Lag_4'] = train_merged.groupby('Store')['Weekly_Sales'].shift(4)

# 4.2 Create Rolling Mean Features to Smooth Data
train_merged['Rolling_Mean_4'] = train_merged.groupby('Store')['Weekly_Sales'].transform(lambda x: x.rolling(window=4).mean())

# 4.3 Convert Categorical Features to Numerical (e.g., Store Type)
train_merged = pd.get_dummies(train_merged, columns=['Type'], drop_first=True)

# Drop rows with missing values created by lag features
train_merged = train_merged.dropna()

# Step 5: Model Building and Evaluation
# 5.1 Define Features and Target Variable
features_list = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales_Lag_1', 'Weekly_Sales_Lag_4', 'Rolling_Mean_4', 'Type_B', 'Type_C']
X = train_merged[features_list]
y = train_merged['Weekly_Sales']

# 5.2 Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5.3 Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5.4 Predict and Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# 5.5 Feature Importance Analysis
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()

# Step 6: Visualization and Reporting
# 6.1 Plot Predicted vs. Actual Sales
plt.figure(figsize=(15, 6))
plt.plot(y_test.values, label='Actual Sales', color='blue')
plt.plot(y_pred, label='Predicted Sales', color='red')
plt.title("Predicted vs Actual Sales")
plt.xlabel("Test Set Data Points")
plt.ylabel("Weekly Sales")
plt.legend()
plt.show()

# 6.2 Save Model and Results (Optional)
# Save the model to disk
import joblib
joblib.dump(model, "walmart_sales_forecast_model.pkl")
