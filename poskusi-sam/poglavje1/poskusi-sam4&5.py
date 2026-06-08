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
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1-out.data**2) * out.grad
        out._backward = _backward
        return out
    
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


#funkcije za koncne diference, za testiranje in primerjanje 
def f(a,b,c):
    return (math.tanh((a+b)))*c

def parc_a(a,b,c, h=0.00001):
    return (f(a+h, b,c) - f(a,b,c)) / h

def parc_b(a,b,c, h=0.00001):
    return (f(a, b+h,c) - f(a,b,c)) / h


def parc_c(a,b,c,h=0.00001):
    return (f(a,b,c+h) - f(a,b,c)) / h


    
    
a = Value(1.0, label = "a")
b = Value(0.5, label = "b")
c = Value(2.0, label = "c")

a_num = 1.0
b_num = 0.5
c_num = 2.0

d = a + b 
d.label = 'd'

t = d.tanh()
t.label = 't'

val = t * c
val.label = 'val'

print(val)
#print(val.backward())
val.backward()
print(f"{a.grad}, {b.grad}, {c.grad} ")
print(f"numericne: {parc_a(a_num, b_num, c_num), parc_b(a_num, b_num, c_num), parc_c(a_num, b_num, c_num)}")

# a = Value(2.0, label="a")
# b = Value(-3.0, label="b")
# c = Value(10.0, label="c")
# d = Value(-2.0, label="d")

# e = a*b
# e.label = "e"
# f = e+c
# f.label = "f"
# L = f*d
# L.label = "L"

# print(L.backward())
    