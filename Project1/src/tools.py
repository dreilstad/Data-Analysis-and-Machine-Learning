import numpy as np

np.random.seed(2)

def generateData(N):
    x, y = np.random.uniform(0, 1, size=(2, N))
    return x, y

def frankeFunction(x, y, noise=0.0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) 
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) 
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0, noise, len(x))

def computeDesignMatrix(x, y, degree):
	N = len(x)
	P = int(degree*(degree+3)/2)

	X = np.zeros(shape=(N, P+1))
	X[:,0] = 1.0

	index = 1
	for i in range(1, degree + 1):
		for j in range(i + 1):
			X[:,index] = (x**(i - j)) * (y**j)
			index += 1
	
	return X