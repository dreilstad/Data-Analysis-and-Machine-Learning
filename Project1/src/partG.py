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
        plt.imshow(terrain[380:400, 480:500], cmap='magma')
        plt.xlabel('X')
        plt.ylabel('Y')
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

        mse_ols_test, mse_ols_train, r2_ols = CrossValidation.kFoldCrossValidation(x, y, z, poly_degree, kfolds, 'OLS')
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

def compareOLSRidgeLassoComplexity(x, y, z, noise, poly_degrees, kfolds, lmbda):

    degrees = np.arange(1, poly_degrees + 1)

    MSE_ols_test = np.zeros(poly_degrees)
    MSE_ols_train = np.zeros(poly_degrees)

    MSE_ridge_test = np.zeros(poly_degrees)
    MSE_ridge_train = np.zeros(poly_degrees)

    MSE_lasso_test = np.zeros(poly_degrees)
    MSE_lasso_train = np.zeros(poly_degrees)

    R2_OLS = np.zeros(poly_degrees)
    R2_Ridge = np.zeros(poly_degrees)
    R2_Lasso = np.zeros(poly_degrees)

    i = 0
    for degree in tqdm(degrees):

        mse_ols_test, mse_ols_train, r2_ols = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, 'OLS')
        mse_ridge_test, mse_ridge_train, r2_ridge = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, 'Ridge', lmbda=lmbda)
        mse_lasso_test, mse_lasso_train, r2_lasso = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, 'Lasso', lmbda=lmbda)

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
    Analysis.plot_ols_ridge_lasso(train_scores, 
                                  test_scores,
                                  z.shape[0],  
                                  noise, 
                                  degrees,
                                  'OLS_Ridge_Lasso_comparison_Lambda=' + str(lmbda))
    
    Analysis.plot_ols_ridge_lasso_r2(R2_OLS, 
                                     R2_Ridge, 
                                     R2_Lasso, 
                                     z.shape[0], 
                                     noise, 
                                     degrees, 
                                     'R2_OLS_Ridge_Lasso_comparison_Lambda=' + str(lmbda))


def lambdaVsComplexity(x, y, z, poly_degrees, kfolds, method):

    R2_test_scores = np.zeros((poly_degrees, 13))
    
    lambdas = np.logspace(-11, 1, 13)
    degrees = np.arange(1, poly_degrees + 1)

    i = 0
    for degree in tqdm(degrees):
        j = 0
        for lmbda in lambdas:
            mse, mse_train, r2 = CrossValidation.kFoldCrossValidation(x, y, z, degree, kfolds, method, lmbda=lmbda)
            R2_test_scores[i][j] = r2

            j += 1

        i += 1

    Analysis.plot_lambda_vs_complexity(R2_test_scores, degrees, lambdas, 'lambda_vs_complexity_heatmap')


def partG(terrain_data, noise, poly_degrees, plot, lmbda=0.0, kfolds=10):
    
    terrain = terrain_data[370:400, 470:500]

    x = np.linspace(0,1, terrain.shape[0])
    y = np.linspace(0,1, terrain.shape[1])
    x, y = np.meshgrid(x, y)


    x = x.flatten()
    y = y.flatten()
    z = terrain.flatten()

    print(z.shape)

    if plot == 'lambda_vs_complexity':
        #lambdaVsComplexity(x, y, z, poly_degrees, 10, 'Ridge')
        lambdaVsComplexity(x, y, z, poly_degrees, 10, 'Lasso')
    elif plot == 'compare_all':
        #compareOLSRidgeLassoComplexity(z, y, z, noise, poly_degrees, 10, lmbda)
        compareOLSRidgeLassoLambda(z, y, z, noise, poly_degrees, 10, lmbda)




terrain_file = 'SRTM_data_Norway_2.tif'
terrain = readTerrain(terrain_file, show=False)

noise = float(sys.argv[1])
poly_degrees = int(sys.argv[2])
plot = int(sys.argv[3])
lmbda = int(sys.argv[4])

lambdas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
plots = ['lambda_vs_complexity', 'compare_all']

partG(terrain, noise, poly_degrees, plots[plot], lmbda=lambdas)