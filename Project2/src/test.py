# Importing various packages
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sgd import SGD
import tools
import analysis as an
'''
import numpy as np

c = np.zeros(10)
d = np.ones(10)

print(c - d)
a = 10
b = 99

print(a > 5)
print((a > 5)*b)
print((a > 50)*b)


'''
N = int(sys.argv[1])
noise = float(sys.argv[2])

n_epochs = 100
degree = 5

x, y = tools.generateData(N)
X = tools.computeDesignMatrix(x, y, degree)
z = tools.frankeFunction(x, y, noise=noise)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
X_train, X_test = tools.scale(X_train, X_test)

degrees = np.arange(1, 16)
scores = np.zeros(len(degrees))

for i, degree in enumerate(degrees):
    MODEL = SGD(X_train, z_train, epochs=n_epochs)
    beta = MODEL.fit_with_decay(t0=5, t1=50)
    print("beta:")
    print(beta)

    error = an.MSE(z_test, X_test @ beta)
    print("MSE:")
    print(error)

    scores[i] = error

plt.plot(degrees, scores, )
plt.yscale("log")
plt.legend()
plt.xlabel(r'Degree')
plt.ylabel(r'MSE score')
plt.grid()
plt.tight_layout()
plt.show()

'''