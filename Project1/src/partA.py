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

    degrees = np.arange(1, poly_degree + 1)
    
    for degree in degrees:

        DesignMatrix = computeDesignMatrix(x, y, degree)

        MODEL = Regression(DesignMatrix, z)
        MODEL.splitData(0.2)
        MODEL.scaleData()
        MODEL.fit()
        MODEL.predict()
        MODEL.predict(test=True)

        MSE_train_scores[degree - 1] = Analysis.MSE(MODEL.z_train, MODEL.z_tilde)
        R2_train_scores[degree - 1] = Analysis.R2(MODEL.z_train, MODEL.z_tilde)

        MSE_test_scores[degree - 1] = Analysis.MSE(MODEL.z_test, MODEL.z_predict)
        R2_test_scores[degree - 1] = Analysis.R2(MODEL.z_test, MODEL.z_predict)


        std_beta = np.sqrt(MODEL.beta_coeff_variance())
        confidence_interval = 1.96 * std_beta

        if degree == 5:
            Analysis.plot_confidence_intervals(MODEL.beta, confidence_interval, N, noise, degree, 'conf_intervals_beta')
    
    
    Analysis.plot_mse_vs_complexity(MSE_train_scores, 
                                    MSE_test_scores, 
                                    N, 
                                    noise, 
                                    poly_degree, 
                                    "mse_vs_complexity")
    
    Analysis.plot_r2_vs_complexity(R2_train_scores,
                                   R2_test_scores,
                                   N,
                                   noise,
                                   poly_degree,
                                   "r2_vs_complexity")

N = int(sys.argv[1])
noise = float(sys.argv[2])
degree = int(sys.argv[3])

partA(N, noise, degree)