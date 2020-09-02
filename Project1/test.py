import numpy as np
import sympy as sym

def get_X_poly2D(x,y,deg):
    X = []
    for i in range(deg + 1):
        for n in range(i+1):
            X.append(x**n * y**(i-n))
    X = (np.array(X).T).squeeze()
    return X

'''
N = 10

x = np.random.rand(N)
y = np.random.rand(N)
np.set_printoptions(precision=3)
print(x)
print(y)

X = np.c_[np.ones(N),
		  x,y,
		  x**2,x*y,y**2,
		  x**3,x**2*y,x*y**2,y**3,
		  x**4,x**3*y,x**2*y**2,x*y**3,y**4,
		  x**5,x**4*y,x**3*y**2,x**2*y**3,x*y**4,y**5]
'''
print(X)