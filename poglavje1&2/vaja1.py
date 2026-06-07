def partial_derivative_a(a, b):
    # TODO: vrni vrednost parcialnega odvoda funkcije f(a,b) po spremenljivki a
    return 2 * a + b - 6

def partial_derivative_b(a, b):
    # TODO: vrni vrednost parcialnega odvoda funkcije f(a,b) po spremenljivki b
    return 2 * b + a - 4

def gradient_descent_2d(start_a, start_b, learning_rate, steps):
    a = start_a
    b = start_b
    
    # Python zanka za ponavljanje (število korakov)
    for i in range(steps):
        # TODO: Izračunaj gradient za a in b z uporabo zgornjih funkcij
        grad_a = partial_derivative_a(a, b)
        grad_b = partial_derivative_b(a,b)
        # TODO: Posodobi vrednosti a in b (smer negativnega gradienta)
        a = a - learning_rate * grad_a
        b = b - learning_rate * grad_b
        
    return a, b

# Testiranje
koncni_a, koncni_b = gradient_descent_2d(start_a=0.0, start_b=0.0, learning_rate=0.1, steps=50)
print(f"Minimum najden pri a = {koncni_a:.3f}, b = {koncni_b:.3f}")

# NE RABIS SKRBET Z IF STAVKI AMPAK SAMO NAPISES FORMULO KER FORMULA SAMA ZA VSE POSKRBI