class Value:
    """ Implementirajte potrebno; pričnite z razredom Value s predavanja. """
    pass


def fn1(x1, x2):
    # implementirajte (definicija v README)
    pass


def fn2(x1, x2):
    # implementirajte (definicija v README)
    pass


if __name__ == "__main__":
    x1 = Value(2, label='x1')
    x2 = Value(1, label='x2')

    fn = fn1

    l = fn(x1, x2)

    print("value", l.data)
    l.backward()
    print("dL", l.grad)
    print("dx1", x1.grad)
    print("dx2", x2.grad)

    # debug :)
    # draw_dot(L, format="png").render()