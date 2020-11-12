import numpy as np
import analysis as an
from sgd import SGD
from ols import OrdinaryLeastSquares
from ridge import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(2)


def computeDesignMatrix(x, y, degree):
    '''Function computes the design matrix for a given degree, where the polynomial degree of
       each column increases up to the given degree.

       The  series is on the following form: [1, x, y, x^2, y^2, xy, ...]

    Args:
        x, y (ndarray): vector of N datapoints
        degree (int): max degree
    Returns:
        a 2D matrix containing the given dataset
    '''
    
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

def generateData(N):
    '''Function generates random data of size N.
    
    Args:
        N (int): number of datapoints
    Returns:
        x, y (ndarray): vector of N datapoints
    '''
    x, y = np.random.uniform(0, 1, size=(2, N))
    return x, y

def frankeFunction(x, y, noise=0.0):
    '''Function returns the Franke function for a corresponding dataset. Also adds given noise.

    Args:
        x, y (ndarray): vector of N datapoints
        noise (float): amount of normally distributed noise to be added
    Returns:
        the franke function
    '''

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) 
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) 
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0, noise, len(x))


def scale(X_train, X_test):
    X_train_scaled = X_train[:,1:]
    X_test_scaled = X_test[:,1:]

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit(X_train_scaled)
    X_train_scaled = scaler.transform(X_train_scaled)
    X_test_scaled = scaler.transform(X_test_scaled)

    X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
    X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

    return X_train_scaled, X_test_scaled

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector