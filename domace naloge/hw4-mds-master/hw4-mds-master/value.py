import math

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
    
    def __truediv__(self, other): # self / other
        # Preverimo, če je 'other' število, in ga pretvorimo v Value
        other = other if isinstance(other, Value) else Value(other)
        # self / other je isto kot self * (other**-1)
        return self * (other**-1)

    def __rtruediv__(self, other): # other / self
        # To se pokliče, ko imamo npr. 10 / Value(...)
        other = other if isinstance(other, Value) else Value(other)
        return other * (self**-1)

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