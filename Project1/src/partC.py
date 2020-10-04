import sys
import numpy as np

from regression import Regression
from analysis import Analysis
from resampling import Bootstrap, CrossValidation
from tools import generateData, frankeFunction, computeDesignMatrix

def partC(N, noise, poly_degree, kfolds, plot, bootstraps=10):

    x, y = generateData(N)
    z = frankeFunction(x, y, noise)

    MSE_train_scores = np.zeros(poly_degree)
    MSE_test_scores = np.zeros(poly_degree)

    MSE_train_scores_boot = np.zeros(poly_degree)
    MSE_test_scores_boot = np.zeros(poly_degree)

    for degree in range(1, poly_degree + 1):
        mse, mse_train, r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, 'OLS')
        MSE_test_scores[degree - 1] = mse
        MSE_train_scores[degree - 1] = mse_train

        result = Bootstrap.bootstrap(x, y, z, degree, bootstraps, 'OLS')
        MSE_test_scores_boot[degree - 1] = result[0]
        MSE_train_scores_boot[degree - 1] = result[1]
    

    if plot == 'kfold_vs_bootstrap':
        Analysis.plot_bootstrap_vs_kfold(MSE_test_scores,
                                         MSE_test_scores_boot,
                                         N,
                                         noise,
                                         poly_degree,
                                         "kfold_vs_bootstrap_Bootstraps=" + str(bootstraps) + 
                                         ",KFolds=" + str(kfolds))

    elif plot == 'mse_vs_complexity':

        Analysis.plot_mse_vs_complexity(MSE_train_scores,
                                        MSE_test_scores,
                                        N,
                                        noise,
                                        poly_degree,
                                        "kfold_mse_vs_complexity_KFolds=" + str(kfolds))

        Analysis.plot_mse_vs_complexity(MSE_train_scores_boot,
                                        MSE_test_scores_boot,
                                        N,
                                        noise,
                                        poly_degree,
                                        "kfold_mse_vs_complexity_Bootstraps=" + str(bootstraps))
                                        

        
N = int(sys.argv[1])
noise = float(sys.argv[2])
degree = int(sys.argv[3])
kfolds = int(sys.argv[4])
plot = int(sys.argv[5])

plots = ['mse_vs_complexity', 'kfold_vs_bootstrap']

partC(N, noise, degree, kfolds, plots[plot])