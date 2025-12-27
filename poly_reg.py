import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# data
X = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
y = np.array([2,5,10,17,26,37,50,65])

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# model: degree 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# evaluation
train_pred = model.predict(X_train_poly)
test_pred = model.predict(X_test_poly)

print("Train MSE:", mean_squared_error(y_train, train_pred))
print("Test MSE:", mean_squared_error(y_test, test_pred))
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# graph
xx = np.linspace(1,8,100).reshape(-1,1)
xx_poly = poly.transform(xx)
yy = model.predict(xx_poly)

plt.scatter(X, y)
plt.plot(xx, yy, color="red")
plt.title("Polynomial Regression (degree=2)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#Train MSE: 4.2335535246860965e-29
#Test MSE: 1.072850831100576e-28
#Model Coefficients: [ 0.00000000e+00 -1.44328993e-15  1.00000000e+00]
#Intercept: 0.9999999999999964