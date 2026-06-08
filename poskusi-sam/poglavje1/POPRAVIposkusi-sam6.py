import math
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
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
        out = Value(self.data + other.data, (self, other), "+")            
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")         
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Value(self.data ** other, (self, ), f"**{other}")
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
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
    
    def backward(self):
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            
            build_topo(self) # Tukaj kličeš funkcijo
            
            self.grad = 1.0
            for v in reversed(topo):
                v._backward()

#funkcije za koncne diference, za testiranje in primerjanje 
def f(a,b):
    return a**2 + b**2 + a*b - 6*a - 4*b

def parc_a(a,b, h=0.00001):
    return (f(a+h, b) - f(a,b)) / h

def parc_b(a,b,h=0.00001):
    return (f(a, b+h) - f(a,b)) / h





def grad_sestop(learning_rate, steps):
    a = Value(0, label="a")
    b = Value(0, label="b")

    for i in range(steps):
        a.grad = 0 
        b.grad = 0
        f = (a**2) + (b**2) + (a*b) - (Value(6.0)*a) - (Value(4.0)*b)
        f.label = "f"

        f.backward()
        a.data -= learning_rate * a.grad
        b.data -= learning_rate * b.grad
    
    return a,b,f


def grad_sestop_analiticno(learning_rate, steps):
    a = 0
    b = 0
    for i in range(steps):
        a -= learning_rate*parc_a(a,b)
        b -= learning_rate*parc_b(a,b)
    
    return a,b

a_analiticno, b_analiticno = grad_sestop_analiticno(0.1, 50)

a_final, b_final, f_final = grad_sestop(0.1, 50)


print(f"a: {a_final.data}, b: {b_final.data}, f: {f_final.data}")
print(f"analiticno: {a_analiticno}, {b_analiticno}, funkcija: {f(a_analiticno, b_analiticno)}")

    
    

    