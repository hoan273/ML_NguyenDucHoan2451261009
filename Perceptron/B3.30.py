import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Perceptron:
    def __init__(self, eta=0.01, n_iter=1000):
        self.eta = eta
        self.n_iter = n_iter
        self.w = None
        self.b = 0

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.n_iter):
            for xi, yi in zip(X, y):
                z = np.dot(self.w, xi) + self.b

                if yi * z <= 0:
                    self.w = self.w + self.eta * yi * xi
                    self.b = self.b + self.eta * yi

        return self

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return np.where(z >= 0, 1, -1)


# 1. Đọc dữ liệu
data = load_breast_cancer()

X = data.data
y = data.target

# Đổi nhãn 0 -> -1, 1 -> 1
y = np.where(y == 0, -1, 1)

# 2. Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Xây dựng và huấn luyện Perceptron
model = Perceptron(eta=0.01, n_iter=1000)
model.fit(X_train, y_train)

# 5. Dự đoán
y_pred = model.predict(X_test)

# 6. Tính các độ đo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("KẾT QUẢ PERCEPTRON")
print("-------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)