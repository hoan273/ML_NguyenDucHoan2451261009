import numpy as np
import matplotlib.pyplot as plt

# Hàm f(x)
def cost(x):
    return x**2 + 2

# Đạo hàm f'(x)
def grad(x):
    return 2*x

# Gradient Descent
def myGD(x0, eta):
    x = [x0]

    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])

        if abs(grad(x_new)) < 1e-3:
            x.append(x_new)
            break

        x.append(x_new)

    return x, it
# Giá trị ban đầu
x0 = 5
eta = 0.01

# Chạy Gradient Descent
x, it = myGD(x0, eta)

print("x =", x[-1])
print("Số lần lặp =", it)
print("f(x) =", cost(x[-1]))


# =========================
# VẼ BIỂU ĐỒ
# =========================

# Tạo các giá trị x để vẽ hàm
x_values = np.linspace(-6, 6, 400)
y_values = cost(x_values)

# Tính giá trị y của các điểm Gradient Descent
y_gd = [cost(i) for i in x]

# Vẽ hàm f(x)
plt.plot(
    x_values,
    y_values,
    color="blue",
    label="f(x) = x² + 2"
)

# Vẽ quá trình Gradient Descent
plt.plot(
    x,
    y_gd,
    color="red",
    marker="o",
    markersize=4,
    label="Gradient Descent"
)

# Điểm bắt đầu
plt.scatter(
    x[0],
    y_gd[0],
    color="green",
    s=100,
    label="Điểm bắt đầu"
)

# Điểm cực tiểu
plt.scatter(
    x[-1],
    y_gd[-1],
    color="blue",
    s=100,
    label="Điểm cực tiểu"
)

# Ghi chú điểm bắt đầu
plt.annotate(
    "Bắt đầu (x₀ = 5)",
    xy=(x[0], y_gd[0]),
    xytext=(3.2, 27),
    arrowprops=dict(arrowstyle="->")
)

# Ghi chú điểm cực tiểu
plt.annotate(
    "Điểm cực tiểu (x = 0, f(x) = 2)",
    xy=(x[-1], y_gd[-1]),
    xytext=(1.2, 5),
    arrowprops=dict(arrowstyle="->")
)

# Tên trục
plt.xlabel("x")
plt.ylabel("f(x)")

# Tiêu đề
plt.title("Gradient Descent tìm điểm cực tiểu")

# Giới hạn biểu đồ
plt.xlim(-6, 6)
plt.ylim(0, 30)

plt.grid()
plt.legend()

plt.show()