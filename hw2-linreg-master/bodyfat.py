import random
import math
import pandas as pd
import numpy as np
class Value:
    """ Implementirajte potrebno; pričnite z razredom Value s predavanja. """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value({self.label}: {self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
                self.grad += 1.0 * out.grad
                other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        out = Value(self.data ** other, (self, ), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def ln(self):
        out = Value(math.log(self.data), (self, ), f'ln{self}')
        def _backward():
            self.grad += (self.data ** (-1)) * out.grad 
        out._backward = _backward
        return out
    
    def e(self):
        out = Value(math.e ** self.data, (self,), f'e^({self.label})')
        
        def _backward():

            self.grad += out.data * out.grad
            
        out._backward = _backward
        return out

    def __radd__(self, other): # other + self
        return self + other

    def __neg__(self): # - self
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other
    
    def __rpow__(self, other):
        return self ** other

    def backward(self):
        # topological ordering of the nodes
        topo = []
        visited = set()
        def build_topo(v):
            v.grad = 0
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # application of chain rule
        self.grad = 1
        for v in reversed(topo):
            v._backward()



class LinReg:
    #pass  # take it from the lectures
    def __init__(self, n_inputs):
        self.weights = [Value(random.uniform(-1,1), label=f'w{i}')
            for i in range(n_inputs)]
        self.b = Value(0.0, label='b')


    def __call__(self, x): #klic modela 
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.b
    
    def parameters(self):
        return self.weights + [self.b]
    
    def loss(self, xs, ys):
        yhats = [self(x) for x in xs]
        n = len(ys)
        return sum([(y - yhat)**2 for y, yhat in zip(ys, yhats)]) * (n**(-1))
    
    def __repr__(self):
        weights_str = ', '.join(f'w{i}={w.data:.3f}' for i, w in enumerate(self.weights))
        return f"LinReg({weights_str}, b={self.b.data:.3f})"

    def batch_loss(self, xs, ys, m=10):
        indices = random.sample(range(len(xs)), m)
        batch_xs = [xs[idx] for idx in indices]
        batch_ys = [ys[idx] for idx in indices]
        return self.loss(batch_xs, batch_ys)
    

def train(model, xs, ys, learning_rate=0.001, n_epochs=1000, batch_size=10):
    for k in range(n_epochs):
        # compute loss
        loss = model.batch_loss(xs, ys, batch_size)

        # compute gradients
        for p in model.parameters():
            p.grad = 0

        loss.backward()
        
        # update
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        #if k % 50 == 0:
            #print(f"{k:3} Loss: {loss.data:5.3f} {model}")
    return model
   



def model_all(batch_size=252):
    df = pd.read_csv("body-fat-brozek.csv")
    feature_names = df.columns[1:].tolist()
    ys = df.iloc[:, 0].tolist()
    X = df.iloc[:, 1:].values.tolist()

    #noramlizacija
    X_arr = np.array(X)
    mean = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    X_norm = ((X_arr - mean) / std).tolist()


    lin = LinReg(n_inputs=len(feature_names))
    model = train(lin, X_norm, ys, n_epochs=1000, batch_size=batch_size, learning_rate=0.1)
    return model, X_norm, ys


def model_wh(batch_size=252):
    df = pd.read_csv("body-fat-brozek.csv")
    # TAKO SE V PANDAS BERE WEIGHT AND HEIGHT
    X_df = df[["body fat brozek","weight", "height"]] 
    feature_names = X_df.columns[1:].tolist()

    ys = X_df.iloc[:, 0].tolist()
    X = X_df.iloc[:, 1:].values.tolist()

    #noramlizacija
    X_arr = np.array(X)
    mean = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    X_norm = ((X_arr - mean) / std).tolist()


    
    lin = LinReg(n_inputs=len(feature_names))
    model = train(lin, X_norm, ys, n_epochs=1000, batch_size=batch_size, learning_rate=0.1)
    return model, X_norm, ys


def model_wh_squared(batch_size=252):
    df = pd.read_csv("body-fat-brozek.csv")
    # TAKO SE V PANDAS BERE WEIGHT AND HEIGHT
    X_df = df[["body fat brozek","weight", "height"]] 
    feature_names = X_df.columns[1:].tolist()

    ys = X_df.iloc[:, 0].tolist()
    X = X_df.iloc[:, 1:].values.tolist()

    feature_names.append("w2")
    feature_names.append("h2")

    for i in X:
        sqr1 = i[0]**2
        sqr2 = i[1]**2
        i.append(sqr1)
        i.append(sqr2)


    #noramlizacija
    X_arr = np.array(X)
    mean = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    X_norm = ((X_arr - mean) / std).tolist()
    
    lin = LinReg(n_inputs=len(feature_names))
    model = train(lin, X_norm, ys, n_epochs=1000, batch_size=batch_size, learning_rate=0.1)
    loss = lin.loss(X_norm, ys).data
    print(loss)
    return model, X_norm, ys






def model_wh_squared_remove_row(batch_size=251, row=41):
    pass  # implement


if __name__ == "__main__":
    # for trying-out the code
    lr = model_wh_squared()
    # ...
