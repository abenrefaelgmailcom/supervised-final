import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


'''1. Building the data for regression'''

X = np.array([0.1, 0.3, 0.5, 0.8, 1.1, 1.5, 1.9, 2.3, 2.8, 3.3, 3.7, 4.2]).reshape(-1, 1)
y = np.array([1.0, 1.3, 1.5, 2.0, 2.4, 2.9, 3.3, 3.8, 4.4, 5.0, 5.4, 6.0])


'''2. KNN Regression Model with a Fixed K'''

k = 3  # Number of neighbors
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X, y)

'''Prediction on a new point'''
new_x = np.array([[3.0]])
new_y_pred = model.predict(new_x)
print(f"Prediction for x=3.0 with k={k}: {new_y_pred[0]:.3f}")

'''Prediction on all the data (training) for performance calculation'''
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"MSE (train): {mse:.4f}")
print(f"R²  (train): {r2:.4f}")



''' 3. Elbow: K versus MSE and R² on all the data'''

K_values = [2, 3, 4, 5, 6]
mse_list = []
r2_list = []

for k in K_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, y)

    y_pred = model.predict(X)

    mse_k = mean_squared_error(y, y_pred)
    r2_k = r2_score(y, y_pred)

    mse_list.append(mse_k)
    r2_list.append(r2_k)

    print(f"K={k} | MSE={mse_k:.4f} | R²={r2_k:.4f}")


''' elbow chart'''
plt.figure(figsize=(8, 5))
plt.plot(K_values, mse_list, marker='o', label='MSE')
plt.plot(K_values, r2_list, marker='s', label='R²')
plt.xlabel('K')
plt.ylabel('Score')
plt.title('K vs MSE / R² (KNN Regression)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

