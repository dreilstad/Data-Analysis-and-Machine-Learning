import numpy as np
import sympy as sym

'''
np.random.seed(2020)
N = 10

x = np.random.rand(N)
y = np.random.rand(N)

print(x)
print(y)

x = np.random.rand(N)
y = np.random.rand(N)

print(x)
print(y)


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
'''
N = 10
x = np.eye(N)
randomIndices = np.random.randint(0, N, N)
y = x[randomIndices,:]
print(x)
print("")
print(y)

