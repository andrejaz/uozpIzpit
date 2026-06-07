def get_gradients(X, ys, weights, b):
    n = len(ys)
    grad_weights = [0.0] * len(weights)
    grad_b = 0.0
    
    for i in range(len(X)):
        # Izračun napovedi za ta primer
        y_hat = b
        for j in range(len(weights)):
            y_hat += weights[j] * X[i][j]
        
        # Izračun napake
        error = y_hat - ys[i]
        
        # Gradienti
        for j in range(len(weights)):
            grad_weights[j] += error * X[i][j]
        grad_b += error

    # Končno povprečenje (pomnoženo z 2)
    for j in range(len(grad_weights)):
        grad_weights[j] = (grad_weights[j] * 2) / n
    grad_b = (grad_b * 2) / n
    
    return grad_weights, grad_b

def train_multivariate(X, ys, learning_rate=0.01, steps=100):
    weights = [0.0] * len(X[0])
    b = 0.0
    
    # TODO: Implementiraj zanko za učenje
    for i in range(steps):
        grad_ws, grad_b = get_gradients(X, ys, weights, b)

        for j in range(len(grad_ws)):
            weights[j] = weights[j] - learning_rate * grad_ws[j]
        
        b -= learning_rate * grad_b
    
    return weights, b


# 2 atributa, cilj je y = 2*x1 + 3*x2 + 1
X = [[1.0, 2.0], [2.0, 1.0], [3.0, 0.0], [1.0, 1.0]]
ys = [7.0, 8.0, 7.0, 6.0]

weights, b = train_multivariate(X, ys, learning_rate=0.05, steps=500)
print(f"w: {[round(w, 2) for w in weights]}, b: {round(b, 2)}")