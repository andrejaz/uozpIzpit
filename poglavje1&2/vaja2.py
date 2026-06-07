import math

def objective(x):
    return (
        math.sin(1.7 * x)
        + 0.4 * math.cos(3.1 * x)
        + 0.15 * (x - 1.5) ** 2
        + math.tanh(x - 0.5)
    )

def numerical_derivative(x, h=0.0001):
    # TODO: Implementiraj metodo končnih diferenc za funkcijo `objective`
    # Uporabi podani korak `h`
    return (objective(x + h) - objective(x)) / h

def find_minimum(start, learning_rate, steps, h=0.0001):
    x = start
    
    # TODO: Implementiraj zanko za gradientni sestop
    # V vsakem koraku oceni odvod z uporabo funkcije `numerical_derivative`
    # Nato posodobi parameter `x`
    for i in range(steps):
        grad = numerical_derivative(x,h)

        new_x = x - learning_rate * grad
        x = new_x    
    return x

# Testiranje iz tvojih primerov
result_grad = numerical_derivative(0.0)
print(f"Numerični odvod v točki 0.0: {result_grad:.3f} (Pričakovano: 2.036)")

# Preizkusi še sam gradientni sestop (začni pri x=0.0, learning_rate=0.1, npr. 50 korakov)
result_min = find_minimum(start=0.0, learning_rate=0.1, steps=50)
print(f"Lokalni minimum najden pri x: {result_min:.3f}")