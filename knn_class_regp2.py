import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


'''1. the data'''

X = np.array([0.1, 0.3, 0.5, 0.8, 1.1, 1.5, 1.9, 2.3, 2.8, 3.3, 3.7, 4.2]).reshape(-1, 1)
y = np.array([1.0, 1.3, 1.5, 2.0, 2.4, 2.9, 3.3, 3.8, 4.4, 5.0, 5.4, 6.0])


'''2. Train/Test split '''

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,      # 25% for Test (3 examples)
    random_state=42,     # To restore the result
)

print("Train size:", len(X_train))
print("Test size :", len(X_test))

'''3. KNN model with a specific K (for example K=3)'''
k = 3
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, y_train)

'''Prediction on the training set (just to see)'''
y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
print(f"[Train] K={k} | MSE={mse_train:.4f} | R²={r2_train:.4f}")

'''Prediction on the test set – this is where the true quality is measured'''
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"[Test]  K={k} | MSE={mse_test:.4f} | R²={r2_test:.4f}")

'''4. Elbow with Train/Test – choose K based on Test performance'''
K_values = [2, 3, 4, 5, 6]
mse_test_list = []
r2_test_list = []

for k in K_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    mse_k = mean_squared_error(y_test, y_test_pred)
    r2_k = r2_score(y_test, y_test_pred)

    mse_test_list.append(mse_k)
    r2_test_list.append(r2_k)

    print(f"[Test] K={k} | MSE={mse_k:.4f} | R²={r2_k:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(K_values, mse_test_list, marker='o', label='MSE (Test)')
plt.plot(K_values, r2_test_list, marker='s', label='R² (Test)')
plt.xlabel('K')
plt.ylabel('Score')
plt.title('K vs MSE / R² on Test Set (KNN Regression)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
