import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# Task 1: Read data and split into training, validation, and testing sets
data = pd.read_csv('data_reg.csv')
X = data[['x1', 'x2']].values
y = data['y'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Plotting the data
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='r', marker='o', label='Training Set')
ax.scatter(X_val[:, 0], X_val[:, 1], y_val, c='g', marker='o', label='Validation Set')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='b', marker='o', label='Testing Set')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.title('Scatter Plot of Training, Validation, and Testing Sets')
plt.show()

# Task 2: Polynomial regression with degrees 1 to 10
degrees = np.arange(1, 11)
mse_degrees = []

fig = plt.figure(figsize=(20, 15))
fig.suptitle('Polynomial Regression Surfaces and Training Examples')

for i, degree in enumerate(degrees, 1):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    ax = fig.add_subplot(3, 4, i, projection='3d')
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='r', marker='o', label='Training Set')

    x1_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    x2_range = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_mesh = np.c_[X1.flatten(), X2.flatten()]
    X_mesh_poly = poly.transform(X_mesh)
    y_mesh = model.predict(X_mesh_poly)
    Y = y_mesh.reshape(X1.shape)

    ax.plot_surface(X1, X2, Y, alpha=0.5, cmap='viridis', label=f'Degree {degree}')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title(f'Degree {degree}')  # Add title with the degree

    # Add some space between the subplots
    ax.view_init(elev=20, azim=(i - 1) * 30)

    # Calculate and store MSE for each degree
    val_pred = model.predict(X_val_poly)
    mse_degrees.append(mean_squared_error(y_val, val_pred))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot validation error vs polynomial degree
plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_degrees, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error on Validation Set')
plt.title('Validation Error vs Polynomial Degree')
plt.show()

# Task 3: Ridge regression with degree 8 and different regularization parameters
degree = 8
alpha_values = [0.001, 0.005, 0.01, 0.1, 10]
mse_values = []

for alpha in alpha_values:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_poly, y_train)

    val_pred_ridge = ridge_model.predict(X_val_poly)
    mse_values.append(mean_squared_error(y_val, val_pred_ridge))

# Plot MSE on validation vs regularization parameter for Ridge regression
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, mse_values, marker='o')
plt.xscale('log')
plt.xlabel('Regularization Parameter (alpha)')
plt.ylabel('Mean Squared Error on Validation Set')
plt.title('Validation Error vs Regularization Parameter (Ridge Regression)')
plt.show()
