import sys
import numpy as np
import seaborn as sns

from regression import Regression
from ridge import Ridge
from lasso import Lasso
from analysis import Analysis
from resampling import Bootstrap, CrossValidation
from tools import generateData, frankeFunction, computeDesignMatrix

def lambdaVsComplexity(x, y, z, poly_degrees, bootstraps):

    R2_test_scores = np.zeros((poly_degrees, poly_degrees))
    
    lambdas = np.logspace(-11, -2, 10)
    degrees = np.arange(1, poly_degrees + 1)

    i = 0
    for degree in degrees:
        j = 0
        for lmbda in lambdas:
            result = Bootstrap.bootstrap(x, y, z, degree, bootstraps, 'Lasso', lmbda=lmbda)
            R2 = result[2]
            R2_test_scores[i][j] = R2

            j += 1

        i += 1

    Analysis.plot_lambda_vs_complexity(R2_test_scores, degrees, lambdas, 'lambda_vs_complexity_heatmap')

def betaConfidenceIntervalsVsLambda(x, y, z, poly_degrees):
    lambdas = np.logspace(-10, -6, 4)

    p = int(poly_degrees*(poly_degrees+3)/2)
    Betas = np.zeros((p + 1, 4))
    Confidence_intervals = np.zeros((p + 1, 4))

    i = 0
    for lmbda in lambdas:

        DesignMatrix = computeDesignMatrix(x, y, poly_degrees)
        MODEL = Lasso(DesignMatrix, z, lmbda)
        MODEL.scaleData()

        beta = MODEL.fit(MODEL.X, MODEL.z)
        MODEL.predict(test=True)

        std_beta = np.sqrt(MODEL.beta_coeff_variance())
        confidence_interval = 1.96 * std_beta

        Betas[:,i] = beta
        Confidence_intervals[:,i] = confidence_interval

        i += 1

    Analysis.plot_beta_ci_vs_lambda(Betas, Confidence_intervals, lambdas)

def biasVariance(x, y, z, noise, poly_degrees, bootstraps, lmbda):

    MSE_test_scores = np.zeros(poly_degrees)
    Bias = np.zeros(poly_degrees)
    Variance = np.zeros(poly_degrees)
    
    for degree in range(1, poly_degrees + 1):

        result = Bootstrap.bootstrap(x, y, z, degree, bootstraps, 'Lasso', lmbda=lmbda)
        MSE_test_scores[degree - 1] = result[0]
        Bias[degree - 1] = result[3]
        Variance[degree - 1] = result[4]


    Analysis.plot_error_bias_variance_vs_complexity(MSE_test_scores, 
                                                    Bias, 
                                                    Variance, 
                                                    N, 
                                                    noise, 
                                                    poly_degrees, 
                                                    'bias_variance_tradeoff_Lambda=' + str(lmbda) + 
                                                    '_Bootstraps=' + str(bootstraps))

def crossValidation(x, y, z, noise, poly_degrees, bootstraps, kfolds, lmbda, compare=False):


    MSE_train_scores = np.zeros(poly_degrees)
    MSE_test_scores = np.zeros(poly_degrees)
    MSE_test_scores_boot = np.zeros(poly_degrees)

    for degree in range(1, poly_degrees + 1):
        mse, mse_train, r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, 'Lasso', lmbda=lmbda)
        MSE_test_scores[degree - 1] = mse
        MSE_train_scores[degree - 1] = mse_train

        if compare:
            result = Bootstrap.bootstrap(x, y, z, degree, bootstraps, 'Lasso', lmbda=lmbda)
            MSE_test_scores_boot[degree - 1] = result[0]
    

    if compare:
        Analysis.plot_bootstrap_vs_kfold(MSE_test_scores,
                                        MSE_test_scores_boot,
                                        N,
                                        noise,
                                        poly_degrees,
                                        "kfold_vs_bootstrap_Bootstraps=" + str(bootstraps) + 
                                        ",KFolds=" + str(kfolds))

    else:
        Analysis.plot_mse_vs_complexity(MSE_train_scores,
                                        MSE_test_scores,
                                        N,
                                        noise,
                                        poly_degrees,
                                        "kfold_mse_vs_complexity_KFolds=" + str(kfolds))

def compareAll(x, y, z, noise, N, poly_degrees, kfolds, lmbda):

    degrees = np.arange(1, poly_degrees + 1)

    MSE_ols_test = np.zeros(poly_degrees)
    MSE_ols_train = np.zeros(poly_degrees)

    MSE_ridge_test = np.zeros(poly_degrees)
    MSE_ridge_train = np.zeros(poly_degrees)

    MSE_lasso_test = np.zeros(poly_degrees)
    MSE_lasso_train = np.zeros(poly_degrees)

    for degree in degrees:

        ols_test, ols_train, ols_r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, 'OLS')
        ridge_test, ridge_train, ridge_r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, 'Ridge', lmbda=lmbda) 
        lasso_test, lasso_train, lasso_r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, 'Lasso', lmbda=lmbda)

        MSE_ols_test[degree - 1] = ols_test
        MSE_ols_train[degree - 1] = ols_train

        MSE_ridge_test[degree - 1] = ridge_test
        MSE_ridge_train[degree - 1] = ridge_train

        MSE_lasso_test[degree - 1] = lasso_test
        MSE_lasso_train[degree - 1] = lasso_train
    

    train_scores = [MSE_ols_train, MSE_ridge_train, MSE_lasso_train]
    test_scores = [MSE_ols_test, MSE_ridge_test, MSE_lasso_test]
    Analysis.plot_ols_ridge_lasso(train_scores, 
                                  test_scores, 
                                  N, 
                                  noise, 
                                  degrees,
                                  'OLS_Ridge_Lasso_comparison_Lambda=' + str(lmbda))




def partE(N, noise, poly_degrees, plot, bootstraps, lmbda=0.0):

    x, y = generateData(N)
    z = frankeFunction(x, y, noise=noise)


    if plot == 'lambda_vs_complexity':
        lambdaVsComplexity(x, y, z, poly_degrees, bootstraps)

    elif plot == 'beta_ci_vs_lambda':
        betaConfidenceIntervalsVsLambda(x, y, z, poly_degrees)
        
    elif plot == 'lasso_bias_variance':
        biasVariance(x, y, z, noise, poly_degrees, bootstraps, lmbda)

    elif plot == 'lasso_cross_validation':
        #crossValidation(x, y, z, noise, poly_degrees, bootstraps, kfolds, lmbda)
        crossValidation(x, y, z, noise, poly_degrees, 50, 10, lmbda, compare=True)
    elif plot == 'compare_all':
        compareAll(x, y, z, noise, N, poly_degrees, 10, lmbda)



N = int(sys.argv[1])
noise = float(sys.argv[2])
degree = int(sys.argv[3])
plot = int(sys.argv[4])
lmbda = int(sys.argv[5])

lambdas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
plots = ['lambda_vs_complexity', 'beta_ci_vs_lambda', 'lasso_bias_variance', 'lasso_cross_validation', 'compare_all']

#partE(N, noise, degree, plots[plot], bootstraps=bootstraps_or_kfolds)
partE(N, noise, degree, plots[plot], 50, lmbda=lambdas[lmbda])