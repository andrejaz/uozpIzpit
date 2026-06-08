import math


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
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
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

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

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

    def tanh(self):
        """Vrne novo vozlišče z vrednostjo tanh(self.data).
        (Za to nalogo brez _backward.)"""
        out = Value(math.tanh(self.data), (self,), "tanh")
        return out


if __name__ == "__main__":
    a = Value(1.0, label="a")
    b = Value(0.5, label="b")
    c = Value(2.0, label="c")
    out = (a + b).tanh() * c
    print(out.data)
    print(out._prev)
