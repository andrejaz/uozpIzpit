def derivative(a):
    """
    Compute the derivative of f(a) = a**2 - 10*a + 28 at a.
    """
    # TODO: implement this function
    return 2 * a - 10


def find_minimum(start, learning_rate, steps):
    """
    Use gradient descent to approximate the minimum of the quadratic function.
    """
    # TODO: implement this function
    a = start
    for i in range(steps):
        grad = derivative(a)
        a -= learning_rate * grad

    return a


if __name__ == "__main__":
    # Simple example for manual testing
    result = find_minimum(6, 0.1, 3)
    print(result)
