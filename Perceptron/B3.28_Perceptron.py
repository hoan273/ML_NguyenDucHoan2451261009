import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y = 1

# 1. Kiểm tra phân lớp
wx = np.dot(w, x)
y_pred = 1 if wx >= 0 else -1

print("w^T x trước cập nhật =", wx)
print("Nhãn dự đoán =", y_pred)
print("Nhãn thực tế =", y)

if y_pred != y:
    print("Mẫu bị phân lớp sai")

    # 2. Cập nhật Perceptron
    w = w + y * x

    print("w sau cập nhật =", w)

    # 3. Tính lại w^T x
    wx_new = np.dot(w, x)
    print("w^T x sau cập nhật =", wx_new)
else:
    print("Mẫu được phân lớp đúng")