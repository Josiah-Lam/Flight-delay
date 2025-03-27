# import modules 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV data
df = pd.read_csv('../cleaning-preprocessing/cleaned_flight_data_with_target.csv')
df.head(5)

# Get Target Variable
X = df.drop(columns=['DEP_DELAY'])  # Feature matrix
y = df['DEP_DELAY'].values  # Target variable
y = y.reshape(len(y), 1)

# Feature Selection
X = X.applymap(lambda x: max(x, 0))
X = X.fillna(X.mean())

from sklearn.feature_selection import chi2, SelectKBest 
# Perform Chi-Square test 
chi2_selector = SelectKBest(score_func=chi2, k=12) 
X_selected = chi2_selector.fit_transform(X, y) # Get selected feature names 
selected_features = X.columns[chi2_selector.get_support()] 
print("Selected Features:", selected_features.tolist())

chi2_scores = pd.Series(chi2_selector.scores_, index=X.columns)
sorted_chi2_scores = chi2_scores.sort_values(ascending=False) 
print(chi2_scores.sort_values(ascending=False))

# Split Data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size = 0.25, random_state = 0)

# Scaling Data 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Train Data using KNN Regressor
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=10)
regressor.fit(X_train, y_train)

# Get Predicted Value
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
np.set_printoptions(precision=2)

# Get Metrics Result Values
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-Squared (R2): {r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--', label="Perfect Fit")

plt.xlabel("Actual Flight Delay")
plt.ylabel("Predicted Flight Delay")
plt.title("Actual vs Predicted Flight Delay (KNN Regression)")
plt.legend()
plt.grid(True)
plt.show()