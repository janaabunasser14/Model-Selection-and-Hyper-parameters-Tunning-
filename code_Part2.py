import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Ignore specific warnings
warnings.filterwarnings("ignore", message="No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored")
warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan.")

# Load training and testing data
train_data = pd.read_csv('train_cls.csv')
test_data = pd.read_csv('test_cls.csv')

# Extract features and labels
X_train = train_data[['x1', 'x2']].values
y_train = train_data['class'].values
X_test = test_data[['x1', 'x2']].values
y_test = test_data['class'].values

# Convert class labels to integers
class_mapping = {'C1': 0, 'C2': 1}
y_train = np.array([class_mapping[label] for label in y_train])
y_test = np.array([class_mapping[label] for label in y_test])

# Task 1: Logistic Regression with Linear Decision Boundary
model_linear = LogisticRegression()
model_linear.fit(X_train, y_train)

# Plot decision boundary for linear model on training set
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')

# Plot decision boundary
coef = model_linear.coef_
intercept = model_linear.intercept_
xx, yy = plt.xlim(), plt.ylim()
x_vals = np.linspace(xx[0], xx[1], 100)
y_vals = -(coef[0][0] * x_vals + intercept) / coef[0][1]
plt.plot(x_vals, y_vals, 'k--')

plt.title('Logistic Regression - Linear Decision Boundary (Training Set)')
plt.show()

# Plot decision boundary for linear model on testing set
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')

# Plot decision boundary
coef = model_linear.coef_
intercept = model_linear.intercept_
xx, yy = plt.xlim(), plt.ylim()
x_vals = np.linspace(xx[0], xx[1], 100)
y_vals = -(coef[0][0] * x_vals + intercept) / coef[0][1]
plt.plot(x_vals, y_vals, 'k--')

plt.title('Logistic Regression - Linear Decision Boundary (Testing Set)')
plt.show()

# Fit the quadratic logistic regression model
model_quadratic = make_pipeline(PolynomialFeatures(degree=2), LogisticRegression())
model_quadratic.fit(X_train, y_train)

# Plot decision boundary for quadratic model on training set
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
                     np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
Z = model_quadratic.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Adjust levels for contour plot
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='k')

plt.title('Logistic Regression - Quadratic Decision Boundary (Training Set)')
plt.show()

# Plot decision boundary for quadratic model on testing set
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
                     np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
Z = model_quadratic.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Adjust levels for contour plot
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='k')

plt.title('Logistic Regression - Quadratic Decision Boundary (Testing Set)')
plt.show()

# Evaluate performance on training set
train_acc_linear = accuracy_score(y_train, model_linear.predict(X_train))

# Evaluate performance on testing set
test_acc_linear = accuracy_score(y_test, model_linear.predict(X_test))

print("Linear Model:")
print(" - Training Accuracy: {:.2f}".format(train_acc_linear))
print(" - Testing Accuracy: {:.2f}".format(test_acc_linear))

# Evaluate performance on training set
train_acc_quadratic = accuracy_score(y_train, model_quadratic.predict(X_train))

# Evaluate performance on testing set
test_acc_quadratic = accuracy_score(y_test, model_quadratic.predict(X_test))

print("\nQuadratic Model:")
print(" - Training Accuracy: {:.2f}".format(train_acc_quadratic))
print(" - Testing Accuracy: {:.2f}".format(test_acc_quadratic))


