import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import analysis as an
import tools as tools
from sgd import SGD
from ols import OrdinaryLeastSquares
from ridge import Ridge


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def compare_with_project1():

    for degree in degrees:

        x, y = tools.generateData(N)
        X = tools.computeDesignMatrix(x, y, degree)
        z = tools.frankeFunction(x, y, noise=noise)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        X_train, X_test = tools.scale(X_train, X_test)




if __name__ == "__main__":
    
    
    N = int(sys.argv[1])
    noise = float(sys.argv[2])
    poly_degrees = int(sys.argv[3])
    params = sys.argv[4]
    epochs = int(sys.argv[5])

    compare_with_project1()