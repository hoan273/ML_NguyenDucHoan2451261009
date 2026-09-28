import numpy as np

class Perceptron:
    def __init__(self, eta=0.1, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter
        self.w = None

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            for xi, yi in zip(X, y):
                y_pred = self.predict_one(xi)

                if y_pred != yi:
                    self.w = self.w + self.eta * yi * xi

        return self

    def predict_one(self, x):
        value = np.dot(self.w, x)

        if value >= 0:
            return 1
        else:
            return -1

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


# Dữ liệu mẫu
X = np.array([
    [2, 2, 1],
    [3, 2, 1],
    [4, 3, 1],
    [5, 4, 1]
])

y = np.array([-1, -1, 1, 1])

# Huấn luyện
model = Perceptron(eta=0.1, n_iter=10)
model.fit(X, y)

print("Trọng số sau khi huấn luyện:", model.w)

# Dự đoán dữ liệu mới
X_new = np.array([
    [2, 3, 1],
    [5, 3, 1]
])

y_pred = model.predict(X_new)

print("Nhãn dự đoán:", y_pred)