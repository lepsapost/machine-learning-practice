from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = load_iris()

X = data.data[:, 0].reshape(-1, 1)  # Sepal Length (X)
y = data.data[:, 2]  # Petal Length (y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


plt.scatter(X, y, color='blue')  # Gerçek veri noktaları
sorted_indices = np.argsort(X_test.reshape(-1))
plt.plot(X_test[sorted_indices], y_pred[sorted_indices], color='red')  # Modelin tahmin ettiği doğrusal çizgi
plt.title('Linear Regression: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.show()


r2 = model.score(X_test, y_test)
print(f"Model R^2 skoru: {r2 * 100:.2f}%")
