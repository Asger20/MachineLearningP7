import numpy as np
import scipy as spy;

def test():
    print("Hello World")
    print(np.__version__)
    print(spy.__version__)

test()

def add(x, y):
    return x + y

print (add(1, 1))

def min(x, y):
    return x - y