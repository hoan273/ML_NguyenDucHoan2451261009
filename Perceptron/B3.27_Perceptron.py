import numpy as np

w = np.array([1, 2, -10])
x = np.array([3, 4, 1])

# 1. Tính w^T x
wx = np.dot(w, x)

# 2. Tính nhãn dự đoán
y_pred = 1 if wx >= 0 else -1

# 3. So sánh với nhãn thực tế
y = -1

print("w^T x =", wx)
print("Nhãn dự đoán =", y_pred)

if y_pred != y:
    print("Điểm dữ liệu bị phân lớp sai")
else:
    print("Điểm dữ liệu được phân lớp đúng")