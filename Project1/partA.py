import sys
import numpy as np

from regression import Regression
from analysis import Analysis
from tools import generateData, frankeFunction, computeDesignMatrix

def partA(N, noise, poly_degree):

    MSE_train_scores = np.zeros(poly_degree)
    R2_train_scores = np.zeros(poly_degree)
    #Betas = np.zeros((p, p))

    MSE_test_scores = np.zeros(poly_degree)
    R2_test_scores = np.zeros(poly_degree)

    x, y = generateData(N)
    z = frankeFunction(x, y, noise)

    
    for degree in range(1, poly_degree + 1):
        DesignMatrix = computeDesignMatrix(x, y, degree)


        OLS = Regression(DesignMatrix, z)
        OLS.splitData(0.2)
        OLS.scaleData()
        OLS.fit()
        OLS.predict()
        OLS.predict(test=True)

        MSE_train_scores[degree - 1] = Analysis.MSE(OLS.z_train, OLS.z_tilde)
        R2_train_scores[degree - 1] = Analysis.R2(OLS.z_train, OLS.z_tilde)
        #Betas[:, degree - 1] = OLS.beta

        MSE_test_scores[degree - 1] = Analysis.MSE(OLS.z_test, OLS.z_predict)
        R2_test_scores[degree - 1] = Analysis.R2(OLS.z_test, OLS.z_predict)

        #Analysis.plot_confidence_intervals(OLS, degree)
    
    
    Analysis.plot_mse_vs_complexity(MSE_train_scores, 
                                    MSE_test_scores, 
                                    N, 
                                    noise, 
                                    poly_degree, 
                                    "mse_vs_complexity")

N = int(sys.argv[1])
noise = float(sys.argv[2])
degree = int(sys.argv[3])

partA(N, noise, degree)