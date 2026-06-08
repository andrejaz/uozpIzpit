from value import Value
import random

class LinReg:
    def __init__(self):
        self.w = Value(random.uniform(-1,1), label="w")
        self.b = Value(0.0, label="b")
                       
    def __call__(self, x):
        return self.w * x + self.b

    def parameters(self):
        return [self.w, self.b]

    def loss(self, xs, ys):
        yhats = [self(x) for x in xs]
        # Uporabiš lahko Value(0), da seštevanje poteka pravilno med Value objekti
        loss = Value(0.0, label="loss")
        for y, yhat in zip(ys, yhats):
            diff = Value(y) - yhat
            loss = loss + (diff * diff)
        return loss

    def __repr__(self):
        return f"LinReg(w={self.w.data:.3f}, b={self.b.data:.3f})"
    
def train(model, xs, ys, learning_rate=0.001, n_epochs=1000):
    for k in range(n_epochs):
        # compute loss
        loss = model.loss(xs, ys)
        # compute gradients
        for p in model.parameters():
            p.grad = 0
        loss.backward()
        # update
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        if k % 50 == 0:
            print(f"{k:3} Loss: {loss.data:5.3f} {model}")
    return model


# podatki
random.seed(42)
n = 5
xs = [random.uniform(-5, 5) for _ in range(n)]
ys = [2*x -1 + random.gauss(0, 4) for x in xs]

# print(xs)
# print(ys)

# lr = LinReg()
# print(lr)
# lv = lr.loss(xs, ys)

# print(lv)
# lv.backward()

# model = train(lr, xs, ys, learning_rate=0.01, n_epochs=500)

# print(model)


stopnja_ucenja = [0.001, 0.01, 0.1, 1.0, 10.0]


for i in range(len(stopnja_ucenja)):
    m = train(LinReg(), xs, ys, learning_rate=stopnja_ucenja[i], n_epochs=200)
    print(f"stopnja ucenja: {stopnja_ucenja[i]}: model: {m}")
    print("----------")

print(m)

