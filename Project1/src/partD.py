import sys
import numpy as np
import seaborn as sns

from ridge import Ridge
from analysis import Analysis
from resampling import Bootstrap, CrossValidation
from tools import generateData, frankeFunction, computeDesignMatrix

def lambdaVsComplexity(x, y, z, poly_degrees, noise, kfolds, method):

    lambdas = np.logspace(-11, -2, 10)
    degrees = np.arange(1, poly_degrees + 1)

    R2_test_scores = np.zeros((poly_degrees, len(lambdas)))

    i = 0
    for degree in degrees:
        j = 0
        for lmbda in lambdas:
            mse, mse_train, r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, method, lmbda=lmbda)
            R2_test_scores[i][j] = r2

            j += 1

        i += 1

    Analysis.plot_lambda_vs_complexity(R2_test_scores, 
                                       degrees, 
                                       lambdas, 
                                       len(z), 
                                       noise, 
                                       'lambda_vs_complexity_heatmap_'+ method, 
                                       method)

def betaConfidenceIntervalsVsLambda(x, y, z, poly_degrees):
    lambdas = np.logspace(-10, -6, 4)

    p = int(poly_degrees*(poly_degrees+3)/2)
    Betas = np.zeros((p + 1, 4))
    Confidence_intervals = np.zeros((p + 1, 4))

    i = 0
    for lmbda in lambdas:

        DesignMatrix = computeDesignMatrix(x, y, poly_degrees)
        MODEL = Ridge(DesignMatrix, z, lmbda)
        MODEL.scaleData()

        beta = MODEL.fit(MODEL.X, MODEL.z)
        MODEL.predict(test=True)

        std_beta = np.sqrt(MODEL.beta_coeff_variance())
        confidence_interval = 1.96 * std_beta


        '''
        print(MODEL.beta)
        print(MODEL.beta.shape)
        print(type(MODEL.beta))
        '''
        Betas[:,i] = beta
        Confidence_intervals[:,i] = confidence_interval

        i += 1

    Analysis.plot_beta_ci_vs_lambda(Betas, Confidence_intervals, lambdas)

def biasVariance(x, y, z, noise, poly_degrees, bootstraps, lmbda):

    MSE_test_scores = np.zeros(poly_degrees)
    Bias = np.zeros(poly_degrees)
    Variance = np.zeros(poly_degrees)
    
    for degree in range(1, poly_degrees + 1):

        result = Bootstrap.bootstrap(x, y, z, degree, bootstraps, 'Ridge', lmbda=lmbda)
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
        mse, mse_train, r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, 'Ridge', lmbda=lmbda)
        MSE_test_scores[degree - 1] = mse
        MSE_train_scores[degree - 1] = mse_train

        if compare:
            result = Bootstrap.bootstrap(x, y, z, degree, bootstraps, 'Ridge', lmbda=lmbda)
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

def partD(N, noise, poly_degrees, plot, bootstraps=0, kfolds=0, lmbda=0.0):

    x, y = generateData(N)
    z = frankeFunction(x, y, noise)


    if plot == 'lambda_vs_complexity':
        lambdaVsComplexity(x, y, z, poly_degrees, noise,  10, 'Ridge')
        lambdaVsComplexity(x, y, z, poly_degrees, noise, 10, 'Lasso')

    elif plot == 'beta_ci_vs_lambda':
        betaConfidenceIntervalsVsLambda(x, y, z, poly_degrees)
        
    elif plot == 'ridge_bias_variance':
        biasVariance(x, y, z, noise, poly_degrees, bootstraps, lmbda)

    elif plot == 'ridge_cross_validation':
        #crossValidation(x, y, z, noise, poly_degrees, bootstraps, kfolds, lmbda)
        crossValidation(x, y, z, noise, poly_degrees, 50, kfolds, lmbda, compare=True)


N = int(sys.argv[1])
noise = float(sys.argv[2])
degree = int(sys.argv[3])
bootstraps = int(sys.argv[4])
plot = int(sys.argv[5])
lmbda = int(sys.argv[6])

lambdas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
plots = ['lambda_vs_complexity', 'beta_ci_vs_lambda', 'ridge_bias_variance', 'ridge_cross_validation']
partD(N, noise, degree, plots[plot], bootstraps=bootstraps, lmbda=lambdas[lmbda])