import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from imageio import imread
from regression import Regression
from ridge import Ridge
from lasso import Lasso
from analysis import Analysis
from resampling import Bootstrap, CrossValidation
from tools import generateData, frankeFunction, computeDesignMatrix

def readTerrain(filename, show=False):

    terrain = imread(filename)

    if show:
        plt.title('Terrain over Norway')
        plt.imshow(terrain[300:400, 400:500], cmap='magma')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.imsave("terrain_subset.png", terrain[300:400, 400:500], cmap='magma', format='png')
        plt.show()
    
    return terrain

def compareOLSRidgeLassoLambda(x, y, z, noise, poly_degree, kfolds, lambdas):

    MSE_ols_test = np.zeros(len(lambdas))
    MSE_ols_train = np.zeros(len(lambdas))

    MSE_ridge_test = np.zeros(len(lambdas))
    MSE_ridge_train = np.zeros(len(lambdas))

    MSE_lasso_test = np.zeros(len(lambdas))
    MSE_lasso_train = np.zeros(len(lambdas))

    R2_OLS = np.zeros(len(lambdas))
    R2_Ridge = np.zeros(len(lambdas))
    R2_Lasso = np.zeros(len(lambdas))




    i = 0
    for lmbda in tqdm(lambdas):

        mse_ols_test, mse_ols_train, r2_ols = CrossValidation.kFoldCrossValidation(x, y, z, poly_degree, kfolds, 'Ridge', lmbda=0.0)
        mse_ridge_test, mse_ridge_train, r2_ridge = CrossValidation.kFoldCrossValidation(x, y, z, poly_degree, kfolds, 'Ridge', lmbda=lmbda)
        mse_lasso_test, mse_lasso_train, r2_lasso = CrossValidation.kFoldCrossValidation(x, y, z, poly_degree, kfolds, 'Lasso', lmbda=lmbda)

        MSE_ols_test[i] = mse_ols_test
        MSE_ols_train[i] = mse_ols_train

        MSE_ridge_test[i] = mse_ridge_test
        MSE_ridge_train[i] = mse_ridge_train

        MSE_lasso_test[i] = mse_lasso_test
        MSE_lasso_train[i] = mse_lasso_train

        R2_OLS[i] = r2_ols
        R2_Ridge[i] = r2_ridge
        R2_Lasso[i] = r2_lasso

        i += 1


    train_scores = [MSE_ols_train, MSE_ridge_train, MSE_lasso_train]
    test_scores = [MSE_ols_test, MSE_ridge_test, MSE_lasso_test]
    Analysis.plot_ols_ridge_lasso_lambda(train_scores, 
                                         test_scores,
                                         z.shape[0],
                                         noise,
                                         lambdas,
                                         'OLS_Ridge_Lasso_comparison_Degree=' + str(poly_degree))

def errorVSComplexity(x, y, z, poly_degrees, method, kfolds, lmbda):

    degrees = np.arange(1, poly_degrees + 1)

    MSE_train_scores = np.zeros(poly_degrees)
    MSE_test_scores = np.zeros(poly_degrees)

    i = 0
    for degree in tqdm(degrees):

        mse, mse_train, r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, method, lmbda=lmbda)

        MSE_test_scores[i] = mse
        MSE_train_scores[i] = mse_train
        i += 1

    Analysis.plot_mse_vs_complexity(MSE_train_scores,
                                    MSE_test_scores,
                                    z.shape[0], 
                                    0.0, 
                                    degrees,
                                    method + '_mse_vs_complexity_Lambda=' + str(lmbda))
    

def lambdaVsComplexity(x, y, z, poly_degrees, kfolds, method):

    
    
    lambdas = np.logspace(-11, -2, 10)
    degrees = np.arange(1, poly_degrees + 1)

    R2_test_scores = np.zeros((poly_degrees, len(lambdas)))

    i = 0
    for degree in tqdm(degrees):
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
                                       0.0, 
                                       'lambda_vs_complexity_heatmap_' + method, 
                                       method)

def methodVsComplexity(x, y, z, poly_degrees, method, lmbda=0.0):

    MSE_train_scores = np.zeros(poly_degrees)
    MSE_test_scores = np.zeros(poly_degrees)
    R2_test_scores = np.zeros(poly_degrees)

    degrees = np.arange(1, poly_degrees + 1)
    
    '''
    print(method)
    mse, mse_train, r2 = CrossValidation.kFoldCrossValidation(x, y, z, poly_degrees, 10, method, lmbda=1e-11)
    print("MSE test: " + str(mse))
    print("MSE train: " + str(mse_train))
    print("R2 test: " + str(r2))
    '''

    X = computeDesignMatrix(x, y, poly_degrees)
    MODEL = Lasso(X, z, 1e-11)
    MODEL.fit(*[X, z])
    z_predict = MODEL.lasso.predict(X)
    z_predict = np.reshape(z_predict, (int(np.sqrt(z.shape[0])), int(np.sqrt(z.shape[0]))))

    plt.imshow(z_predict, cmap='magma')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.imsave("terrain_predict_Lasso.png", z_predict, cmap='magma', format='png')
    plt.show()

    return
    
    for degree in tqdm(degrees):
        mse, mse_train, r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, 10, method, lmbda=lmbda)

        MSE_train_scores[degree - 1] = mse_train
        MSE_test_scores[degree - 1] = mse
        R2_test_scores[degree - 1] = r2

    print("Lowest MSE test score: " + str(np.min(MSE_test_scores)))
    print("for polynomial degree: " + str(np.argmin(MSE_test_scores) + 1))

    output_mse = open('output_mse.txt', 'a')
    output_mse.write(method + ': ' + str(MSE_test_scores) + '\n')
    output_mse.close()

    Analysis.plot_mse_vs_complexity(MSE_train_scores, 
                                    MSE_test_scores, 
                                    len(z), 
                                    0.0,
                                    poly_degrees, 
                                    "mse_vs_complexity_" + method)


    
    '''
    Analysis.plot_r2_vs_complexity(R2_train_scores,
                                   R2_test_scores,
                                   N,
                                   noise,
                                   poly_degrees,
                                   "r2_vs_complexity")
    '''

def biasVariance(x, y, z, noise, poly_degrees, method, lmbda):

    MSE_test_scores = np.zeros(poly_degrees)
    Bias = np.zeros(poly_degrees)
    Variance = np.zeros(poly_degrees)
    
    for degree in range(1, poly_degrees + 1):

        result = Bootstrap.bootstrap(x, y, z, degree, 50, method, lmbda=lmbda)
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

def partG(terrain_data, poly_degrees, method, plot, lmbda=0.0):
    
    terrain = terrain_data[300:400, 400:500]

    x = np.linspace(0,1, terrain.shape[0])
    y = np.linspace(0,1, terrain.shape[1])
    x, y = np.meshgrid(x, y)


    x = x.flatten()
    y = y.flatten()
    z = terrain.flatten()
    z = z - np.min(z)
    z = z / np.max(z)

    print(z.shape)

    biasVariance(x, y, z, 0.0, poly_degrees, method, lmbda)
    return
    
    if plot == 'method_vs_complexity':
        methodVsComplexity(x, y, z, poly_degrees, method, lmbda)

    elif plot == 'lambda_vs_complexity':
        lambdaVsComplexity(x, y, z, poly_degrees, 10, method)

    elif plot == 'compare_all':
        #compareOLSRidgeLassoComplexity(z, y, z, noise, poly_degrees, 10, lmbda)
        compareOLSRidgeLassoLambda(z, y, z, poly_degrees, 10, lmbda)

    elif plot == 'mse_vs_complexity':
        errorVSComplexity(x, y, z, poly_degrees, method, 10, lmbda)






terrain_file = 'SRTM_data_Norway_2.tif'
terrain = readTerrain(terrain_file, show=False)

method = sys.argv[1]
poly_degrees = int(sys.argv[2])
plot = int(sys.argv[3])
lmbda = int(sys.argv[4])

lambdas = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
plots = ['method_vs_complexity', 'lambda_vs_complexity', 'compare_all', 'mse_vs_complexity']

#partG(terrain, noise, poly_degrees, plots[plot], lmbda=lambdas[lmbda])
partG(terrain, poly_degrees, method, plots[plot], lmbda=lambdas[lmbda])