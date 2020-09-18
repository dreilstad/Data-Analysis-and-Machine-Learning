import sys
import numpy as np

from regression import Regression
from analysis import Analysis
from resampling import Bootstrap, CrossValidation
from tools import generateData, frankeFunction, computeDesignMatrix

def partB(N, noise, poly_degree, bootstraps, plot):

    MSE_test_scores = np.zeros(poly_degree)
    MSE_train_scores = np.zeros(poly_degree)
    Bias = np.zeros(poly_degree)
    Variance = np.zeros(poly_degree)

    x, y = generateData(N)
    z = frankeFunction(x, y, noise)

    for degree in range(1, poly_degree + 1):
        
        mse, mse_train, bias, variance, beta_average, beta_variance = Bootstrap.bootstrap(x, y, z, degree, bootstraps, 'OLS')
        MSE_test_scores[degree - 1] = mse
        MSE_train_scores[degree - 1] = mse_train
        Bias[degree - 1] = bias
        Variance[degree - 1] = variance


    if plot == 'mse_vs_complexity':
        Analysis.plot_mse_vs_complexity(MSE_train_scores, 
                                        MSE_test_scores, 
                                        N, 
                                        noise, 
                                        poly_degree, 
                                        "bootstrapped_mse_vs_complexity_Bootstraps=" + str(bootstraps))
    elif plot == 'bias_variance':
        Analysis.plot_error_bias_variance_vs_complexity(MSE_test_scores, 
                                                        Bias, 
                                                        Variance, 
                                                        N, 
                                                        noise, 
                                                        poly_degree, 
                                                        "bias_variance_tradeoff_Bootstraps=" + str(bootstraps))


N = int(sys.argv[1])
noise = float(sys.argv[2])
degree = int(sys.argv[3])
bootstraps = int(sys.argv[4])
plot = int(sys.argv[5])

plots = ['mse_vs_complexity', 'bias_variance']

partB(N, noise, degree, bootstraps, plots[plot])