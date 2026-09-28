import numpy as np
def cost(x):
    return (1/3)*(x**3)-x
def grad(x):
    return x**2-1
def myGD(x0, eta):
    x = [x0]

    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])

        if abs(grad(x_new)) < 1e-3:
            x.append(x_new)
            break

        x.append(x_new)

    return x, it
x, it = myGD(0, 0.1)


print("Nghiệm gần đúng:", x[-1])
print("Số lần lặp:", it + 1)
print("Giá trị cực tiểu:", (1/3)*x[-1]**3 - x[-1])
