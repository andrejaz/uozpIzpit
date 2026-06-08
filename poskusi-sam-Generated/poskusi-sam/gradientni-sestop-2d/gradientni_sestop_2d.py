class Value:
    """Vozlišče v računskem grafu za strojno odvajanje."""

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
        out = Value(self.data ** other, (self,), f"**{other}")
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def backward(self):
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
        self.grad = 1
        for v in reversed(topo):
            v._backward()


def f_value(a, b):
    """Vrne vozlišče Value za f(a, b) = a^2 + b^2 + a*b - 6a - 4b."""
    # c = Value(a**2)
    # d = Value(b**2)
    # e = Value(a*b)
    # f = Value(6*a)
    # e = Value(4*b)

    # v1 = c+d
    # v2 = v1 + e
    # v3 = v2 - f
    # v4 = v3 - e
    # return v4
    #return (a**2 + b**2) + a*b - 6*a - 4*b
    return a**2 + b**2 + a*b - 6*a - 4*b


def train_2d(lr=0.1, steps=50, a0=0.0, b0=0.0):
    """Gradientni sestop nad f(a, b). Vrne (a, b, f) kot float."""
    a = Value(a0, label="a")
    b = Value(b0, label="b")
    
    for i in range(steps):
        v = f_value(a,b)
        v.backward()
        a.data -= lr * v.grad
        b.data -= lr * v.grad

    return a.data, b.data, f_value(a,b).data


def analytic_min():
    """Vrne analitično rešitev (a*, b*, f*) = (8/3, 2/3, -28/3)."""
    return 8/3, 2/3, -28/3


if __name__ == "__main__":
    print("analiticno:", analytic_min())
    print("ucenje:    ", train_2d(lr=0.1, steps=200))
