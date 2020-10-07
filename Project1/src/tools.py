import numpy as np
import matplotlib.pyplot as plt

from imageio import imread
from ols import OrdinaryLeastSquares
from ridge import Ridge
from lasso import Lasso
from analysis import Analysis
from resampling import Bootstrap, CrossValidation

np.random.seed(2)

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

def readTerrain(filename, show=False):
    '''Function reads terrain data of given file. Has the option to show the terrain.

    Args:
        filename (string): name of fiel with the terraind data
        show (bool): if True shows the terrain
    Returns:
        the terrain data as a 2D array
    '''

    terrain = imread(filename)

    if show:
        plt.title('Terrain over Norway')
        plt.imshow(terrain, cmap='magma')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.imsave("terrain.png", terrain, cmap='magma', format='png')
        plt.show()
    
    return terrain

def compareAll(x, y, z, noise, poly_degrees, lmbda, kfolds):
    '''Function calculates the MSE test and train error for each of the regression methods as a
       function of the complexity. K-fold cross validation is used as the resampling technique.
       Plots the result.

    Args:
        x, y, z (ndarray): dataset
        noise (float): the amount of noise to be added
        poly_degrees (int): max polynomial degree
        lmbda (float): lambda value
        kfolds (int): number of folds
    '''

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
                                  len(z), 
                                  noise, 
                                  degrees,
                                  'OLS_Ridge_Lasso_comparison_Lambda=' + str(lmbda))

def biasVariance(x, y, z, noise, poly_degrees, method, lmbda, bootstraps):
    '''Function calculates the bias-variance decompostion of a given regression method as a
       function of the complexity. The bootstrap method is used as the resampling technique.
       Plots the result.

    Args:
        x, y, z (ndarray): dataset
        noise (float): the amount of noise to be added
        poly_degrees (int): max polynomial degree
        method (string): name of hte regression method to be used
        lmbda (float): lambda value
        bootstraps (int): number of iterations
    '''

    MSE_test_scores = np.zeros(poly_degrees)
    Bias = np.zeros(poly_degrees)
    Variance = np.zeros(poly_degrees)
    
    degrees = np.arange(1, poly_degrees + 1)

    for degree in degrees:

        result = Bootstrap.bootstrap(x, y, z, degree, bootstraps, method, lmbda=lmbda)
        MSE_test_scores[degree - 1] = result[0]
        Bias[degree - 1] = result[3]
        Variance[degree - 1] = result[4]


    Analysis.plot_error_bias_variance_vs_complexity(MSE_test_scores, 
                                                    Bias, 
                                                    Variance, 
                                                    len(z), 
                                                    noise, 
                                                    poly_degrees, 
                                                    'bias_variance_tradeoff_Lambda=' + str(lmbda) + 
                                                    '_Bootstraps=50')

def crossValidation(x, y, z, noise, poly_degrees, method, lmbda, bootstraps=50, kfolds=10, compare=False):
    '''Function calculates the MSE test and train error for a given regression method as a
       function of the complexity. K-fold cross validation is used as the resampling technique.
       If specified, bootstrap is also used and compared to k-fold cross validation. 
       Plots the result. 

    Args:
        x, y, z (ndarray): dataset
        noise (float): the amount of noise to be added
        poly_degrees (int): max polynomial degree
        method (string): name of hte regression method to be used
        lmbda (float): lambda value
        bootstraps (int): number of iterations
        kfolds (int): number of folds
        compare (bool): if True, the resampling techniques are compared
    '''

    MSE_train_scores = np.zeros(poly_degrees)
    MSE_test_scores = np.zeros(poly_degrees)
    MSE_test_scores_boot = np.zeros(poly_degrees)

    degrees = np.arange(1, poly_degrees + 1)

    for degree in degrees:
        mse, mse_train, r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, method, lmbda=lmbda)
        MSE_test_scores[degree - 1] = mse
        MSE_train_scores[degree - 1] = mse_train

        if compare:
            result = Bootstrap.bootstrap(x, y, z, degree, bootstraps, method, lmbda=lmbda)
            MSE_test_scores_boot[degree - 1] = result[0]
    

    if compare:
        Analysis.plot_bootstrap_vs_kfold(MSE_test_scores,
                                        MSE_test_scores_boot,
                                        len(z),
                                        noise,
                                        poly_degrees,
                                        "kfold_vs_bootstrap_Bootstraps=" + str(bootstraps) + 
                                        ",KFolds=" + str(kfolds))

    else:
        Analysis.plot_mse_vs_complexity(MSE_train_scores,
                                        MSE_test_scores,
                                        len(z),
                                        noise,
                                        poly_degrees,
                                        "kfold_mse_vs_complexity_KFolds=" + str(kfolds))

def errorVsComplexity(x, y, z, noise, poly_degrees, method, lmbda, k, technique='cv'):
    '''Function calculates the MSE test and train error for a given regression method as a
       function of the complexity. The resampling technique is given as an argument.
       Plots the result. 

    Args:
        x, y, z (ndarray): dataset
        noise (float): the amount of noise to be added
        poly_degrees (int): max polynomial degree
        method (string): name of hte regression method to be used
        lmbda (float): lambda value
        k (int): number of iterations/folds
        technique (string): name of resampling method to be used
    '''

    MSE_train_scores = np.zeros(poly_degrees)
    MSE_test_scores = np.zeros(poly_degrees)

    degrees = np.arange(1, poly_degrees + 1)

    for degree in degrees:

        if technique == 'cv':
            mse, mse_train, r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, k, method, lmbda=lmbda)
        else:
            result = Bootstrap.bootstrap(x, y, z, degree, k, method, lmbda=lmbda)
            mse = result[0]
            mse_train = result[1]

        MSE_test_scores[degree - 1] = mse
        MSE_train_scores[degree - 1] = mse_train

    print("Lowest MSE test score: " + str(np.min(MSE_test_scores)))
    print("for polynomial degree: " + str(np.argmin(MSE_test_scores) + 1))

    if technique == 'cv':
        figname = method + '_mse_vs_complexity_Kfolds= ' + str(k) + '_Lambda=' + str(lmbda)
    else:
        figname = method + '_mse_vs_complexity_Bootstraps= ' + str(k) + '_Lambda=' + str(lmbda)

    Analysis.plot_mse_vs_complexity(MSE_train_scores,
                                    MSE_test_scores,
                                    len(z), 
                                    noise, 
                                    poly_degrees,
                                    figname)

def errorVsComplexityNoResampling(x, y, z, noise, poly_degrees, method, lmbda, R2=False, CI=-1):
    '''Function calculates the MSE test and train error for a given regression method as a
       function of the complexity. If specified the R2 score is also calculated and also
       the confidence intervals of the beta coefficents for a given polynomial degree.
       Plots the result. 

    Args:
        x, y, z (ndarray): dataset
        noise (float): the amount of noise to be added
        poly_degrees (int): max polynomial degree
        method (string): name of hte regression method to be used
        lmbda (float): lambda value
        R2 (bool): if True, then the R2 is calculated and plotted
        CI (int): polynomial degree to calculate confidence intervals of the beta coefficients
    '''
    MSE_train_scores = np.zeros(poly_degrees)
    R2_train_scores = np.zeros(poly_degrees)

    MSE_test_scores = np.zeros(poly_degrees)
    R2_test_scores = np.zeros(poly_degrees)

    degrees = np.arange(1, poly_degrees + 1)
    
    for degree in degrees:

        DesignMatrix = computeDesignMatrix(x, y, degree)

        if method == 'OLS':
            MODEL = OrdinaryLeastSquares(DesignMatrix, z)
            lmbda = 0.0
        elif method == 'Ridge':
            MODEL = Ridge(DesignMatrix, z, lmbda)
        elif method == 'Lasso':
            MODEL = Lasso(DesignMatrix, z, lmbda)

        MODEL.splitData(0.2)
        MODEL.scaleData()
        MODEL.fit()
        MODEL.predict()
        MODEL.predict(test=True)

        MSE_train_scores[degree - 1] = Analysis.MSE(MODEL.z_train, MODEL.z_tilde)
        R2_train_scores[degree - 1] = Analysis.R2(MODEL.z_train, MODEL.z_tilde)

        MSE_test_scores[degree - 1] = Analysis.MSE(MODEL.z_test, MODEL.z_predict)
        R2_test_scores[degree - 1] = Analysis.R2(MODEL.z_test, MODEL.z_predict)


        if CI != -1 and CI == degree:
            std_beta = np.sqrt(MODEL.beta_coeff_variance())
            confidence_interval = 1.96 * std_beta
            
            Analysis.plot_confidence_intervals(MODEL.beta, 
                                               confidence_interval, 
                                               len(z), 
                                               noise, 
                                               degree, 
                                               method + '_conf_intervals_beta_Lambda=' + str(lmbda))
    

    Analysis.plot_mse_vs_complexity(MSE_train_scores, 
                                    MSE_test_scores, 
                                    len(z), 
                                    noise, 
                                    poly_degrees, 
                                    method + '_mse_vs_complexity_Lambda=' + str(lmbda))
    
    if R2:
        Analysis.plot_r2_vs_complexity(R2_train_scores,
                                       R2_test_scores,
                                       len(z),
                                       noise,
                                       poly_degrees,
                                       method + '_r2_vs_complexity_Lambda=' + str(lmbda))

def lambdaVsComplexity(x, y, z, noise, poly_degrees, method, kfolds):
    '''Function calculates the R2 score for a given regression method as a
       function of the complexity and lambda value. K-fold cross validation is used 
       as the resampling technique. Plots the result as a heatmap.

    Args:
        x, y, z (ndarray): dataset
        noise (float): the amount of noise to be added
        poly_degrees (int): max polynomial degree
        method (string): name of hte regression method to be used, Ridge or Lasso
        kfolds (int): number of folds
    '''

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
                                       'lambda_vs_complexity_heatmap_' + method, 
                                       method)



