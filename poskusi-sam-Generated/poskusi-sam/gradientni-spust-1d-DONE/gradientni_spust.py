def f(x):
    """Testna funkcija f(x) = x^4 - 3x^3 + 2."""
    return x ** 4 - 3 * x ** 3 + 2


def numerical_derivative(func, x, h=1e-5):
    """Numerični odvod func v točki x (končne diference)."""
    return (func(x+h)-func(x))/h


def gradient_descent(func, x0, lr=0.01, steps=1000, h=1e-5):
    """Z numeričnim odvodom poišče lokalni minimum. Vrne končni x."""
    x = x0
    for i in range(steps):
        x -= lr*numerical_derivative(func, x, h)
    
    return x


if __name__ == "__main__":
    for x0 in [0.5, 1.0, 2.0, 2.9]:
        x = gradient_descent(f, x0, lr=0.01, steps=20000)
        print(f"x0={x0}  ->  x={x:.4f}  f(x)={f(x):.4f}")
