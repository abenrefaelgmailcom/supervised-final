import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------
# 1. Prepare data
# -----------------------
X = np.array([40,50,60,70,80,90,100,110,120,130]).reshape(-1,1)
Y = np.array([100,120,145,170,195,215,240,265,290,320])

# -----------------------
# 2. Fit model
# -----------------------
model = LinearRegression()
model.fit(X, Y)

b0 = model.intercept_
b1 = model.coef_[0]

print("b0 (intercept) =", b0)
print("b1 (slope) =", b1)

# -----------------------
# 3. Evaluation
# -----------------------
Y_pred = model.predict(X)
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print("MSE =", mse)
print("R2 Score =", r2)

# -----------------------
# 4. Draw graph
# -----------------------
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel("House Size (m²)")
plt.ylabel("Price (k$)")
plt.title("Linear Regression — House Prices")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------
# 5. Predict new value X=103
# -----------------------
new_x = np.array([[103]])
prediction_103 = model.predict(new_x)
print("Prediction for X=103 =", prediction_103[0])

# -----------------------
# 6. Solve reverse: What X gives Y=1000?
# -----------------------
target_y = 1000
x_for_1000 = (target_y - b0) / b1
print("X needed for price 1000k$ =", x_for_1000)

#b0 (intercept) = -0.06060606060606233
#b1 (slope) = 2.4242424242424243
#MSE = 5.515151515151506
#R2 Score = 0.9988637924361039


