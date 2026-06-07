def get_gradients(xs, ys, w, b):
    n = len(xs)
    grad_w = 0.0
    grad_b = 0.0
    
    # TODO: Z uporabo zanke pojdi čez vse pare (x, y) v xs in ys
    # Za vsak primer izračunaj napoved y_hat in seštej vrednosti za gradienta
    for i in range(n):
        y_hat = w * xs[i] + b
        grad_w += (y_hat - ys[i]) * xs[i]
        grad_b += (y_hat - ys[i]) 

    grad_b = (grad_b * 2) / n
    grad_w = (grad_w * 2) / n

    
    # TODO: Ne pozabi deliti z n (in pomnožiti z 2, če slediš zgornji formuli)

    
    return grad_w, grad_b

def train_linear_regression(xs, ys, learning_rate=0.01, steps=100):
    # Začnemo z naključnimi ali ničelnimi parametri
    w = 0.0
    b = 0.0
    
    # TODO: Implementiraj zanko za gradientni sestop,
    # v kateri pridobiš gradienta in posodobiš parametra w in b.
    for i in range(steps):
        grad_w, grad_b = get_gradients(xs, ys, w,b)

        new_w = w - learning_rate * grad_w
        w = new_w
        new_b = b - learning_rate * grad_b
        b = new_b
    
    return w, b

# Testni podatki
xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

# Rešitev teh podatkov je točno w=2.0 in b=1.0 (saj je y = 2x + 1)
koncni_w, koncni_b = train_linear_regression(xs, ys, learning_rate=0.05, steps=200)
print(f"Naučeni model: w = {koncni_w:.3f}, b = {koncni_b:.3f}")