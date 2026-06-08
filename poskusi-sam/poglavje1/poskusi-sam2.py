def f(x):
    return x**4 - 3*x**3 + 2

def odvod(x, h=0.0001):
    return (f(x+h)-f(x))/h


def grad_spust(steps, learning_rate):
    x = 1

    for i in range(steps):
        odvd = odvod(x)
        x -= learning_rate*odvd

        x = max(0, min(3,x))
    return x,f(x)


x_min, f_vred = grad_spust(1000, 0.01)
print(f"min x: {x_min}, funkcija vrednost: {f_vred}")
