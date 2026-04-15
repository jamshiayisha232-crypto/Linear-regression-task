# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("dataset.csv")

# Show first 5 rows (optional)
print(data.head())

# Select Features (X) and Target (y)
X = data[['area']]
y = data['price']

# Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict Values
y_pred = model.predict(X_test)

# Evaluate Model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot Graph
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression")
plt.show()
