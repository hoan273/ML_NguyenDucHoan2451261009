def f(x):
    return x**2 - 4*x + 5

def grad(x):
    return 2*x - 4

def myGD(x0, eta):
    x = [x0]

    for i in range(4):
        x_new = x[-1] - eta * grad(x[-1])
        x.append(x_new)

    return x

x = myGD(5, 0.2)

for i in range(len(x)):
    print(f"Bước {i}: x = {x[i]}, f(x) = {f(x[i])}")