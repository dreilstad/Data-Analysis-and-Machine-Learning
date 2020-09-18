import sys
import numpy as np

from regression import Regression
from analysis import Analysis
from resampling import Bootstrap, CrossValidation
from tools import generateData, frankeFunction, computeDesignMatrix

def partD(N, noise, p, bootstraps, N_lambdas, plot):

    x, y = generateData(N)
    z = frankeFunction(x, y, noise)

    if plot = 'lambda_vs_complexity':
        MSE_test_scores = np.zeros(p, N_lambdas)
        
        lambdas = np.logspace(-10, 5)
        degrees = np.arange(1, p)

        i = 0
        for degree in degrees:
            j = 0
            for lmbda in lambdas:
                mse, mse_train, bias, variance, beta_average, beta_variance = bootstrap(x, y, z, degree, bootstraps, lmbda, 'Ridge')



    MSE_test_scores = np.zeros(p, N_lambdas)
    MSE_train_scores = np.zeros(N_lambdas)
    Bias = np.zeros(N_lambdas)
    Variance = np.zeros(N_lambdas)


    i = 0
    for lmbda in lambdas:
        
        mse, mse_train, bias, variance, beta_average, beta_variance = bootstrap(x, y, z, degree, bootstraps, lmbda, 'Ridge')
        MSE_test_scores[i] = mse
        MSE_train_scores[i] = mse_train
        Bias[i] = bias
        Variance[i] = variance

        i += 1


    '''
    if plot == 'mse_vs_lambda':
        Analysis.plot_mse_vs_lambda(lambdas,
                                    MSE_train_scores, 
                                    MSE_test_scores, 
                                    N, 
                                    noise, 
                                    p, 
                                    plot + "_Bootstraps=" + str(bootstraps))
    elif plot == 'ridge_bias_variance':
        Analysis.plot_error_bias_variance_vs_lambda(lambdas,
                                                    MSE_test_scores, 
                                                    Bias, 
                                                    Variance, 
                                                    N, 
                                                    noise, 
                                                    p, 
                                                    "bias_variance_tradeoff_NLambdas=" + str(N_lambdas) +
                                                    "Bootstraps=" + str(bootstraps))
    '''