import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from regression import Regression
from analysis import Analysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyvander2d

PROJECT_ROOT_DIR = "Results" 
FIGURE_ID = "Results/FigureFiles" 
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR): 
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID): 
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID): 
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

np.random.seed(2)

def computeDesignMatrix(x, y, degree):
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

def generateData(N):
    x, y = np.random.uniform(0, 1, size=(2, N))
    return x, y

def addNoise(x, y, noise_strength, N):
    
    noise_amount = np.random.normal(0, noise_strength, N)
    x += noise_amount
    y += noise_amount

    return x, y

def frankeFunction(x, y, noise=0.0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) 
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) 
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0, noise, len(x))

def kFoldValidation(x, y, z, degree, k):

    DesignMatrix = computeDesignMatrix(x, y, degree)
    
    OLS_kfold = 0


def bootstrap(x, y, z, degree, bootstraps):

    DesignMatrix = computeDesignMatrix(x, y, degree)

    OLS_boot = Regression(DesignMatrix, z)
    OLS_boot.splitData(0.2)
    OLS_boot.scaleData()

    betas = np.zeros((OLS_boot.X_train.shape[1], bootstraps))
    z_predicts = np.zeros((OLS_boot.z_test.shape[0], bootstraps))
    z_tildes = np.zeros((OLS_boot.z_train.shape[0], bootstraps))
    z_train_samples = np.zeros((OLS_boot.z_train.shape[0], bootstraps))
    
    samples = OLS_boot.X_train.shape[0]
    for i in range(bootstraps):

        # get random indices
        randomIndices = np.random.randint(0, samples, samples)

        # draw samples
        X_train_boot = OLS_boot.X_train[randomIndices]
        z_train_boot = OLS_boot.z_train[randomIndices]

        # fit and predict
        beta = np.linalg.pinv(X_train_boot) @ z_train_boot
        z_tilde = X_train_boot @ beta
        z_predict = OLS_boot.X_test @ beta

        # add result
        betas[:,i] = beta
        z_predicts[:,i] = z_predict
        z_tildes[:,i] = z_tilde
        z_train_samples[:,i] = z_train_boot
    

    # calculate average and variance for all Beta_0, Beta_1, and so on..
    beta_average = np.average(betas, axis=1)
    beta_variance = np.var(betas, axis=1)

    z_test = OLS_boot.z_test.reshape((len(OLS_boot.z_test), 1))

    mse = np.mean(np.mean((z_test - z_predicts)**2, axis=1, keepdims=True))
    mse_train = np.mean(np.mean((z_tildes - z_train_samples)**2, axis=1, keepdims=True))
    bias = np.mean((z_test - np.mean(z_predicts, axis=1, keepdims=True))**2)
    variance = np.mean(np.var(z_predicts, axis=1, keepdims=True))

    return mse, mse_train, bias, variance, beta_average, beta_variance



def partB(N, noise, p, k, plot_bias_variance):

    MSE_scores = np.zeros(p)
    MSE_train_scores = np.zeros(p)
    Bias = np.zeros(p)
    Variance = np.zeros(p)

    x, y = generateData(N)
    z = frankeFunction(x, y, noise)

    for degree in range(1, p + 1):
        
        mse, mse_train, bias, variance, beta_average, beta_variance = bootstrap(x, y, z, degree, k)
        MSE_scores[degree - 1] = mse
        MSE_train_scores[degree - 1] = mse_train
        Bias[degree - 1] = bias
        Variance[degree - 1] = variance


    if not plot_bias_variance:
        Analysis.plot_mse_vs_complexity(MSE_train_scores, 
                                        MSE_scores, 
                                        N, 
                                        noise, 
                                        p, 
                                        "bootstrapped_mse_vs_complexity_Bootstraps=" + str(k))
    else:
        Analysis.plot_error_bias_variance_vs_complexity(MSE_scores, 
                                                        Bias, 
                                                        Variance, 
                                                        N, 
                                                        noise, 
                                                        p, 
                                                        "bias_variance_tradeoff_Bootstraps=" + str(k))


def partA(N, noise, p):

    MSE_train_scores = np.zeros(p)
    R2_train_scores = np.zeros(p)
    #Betas = np.zeros((p, p))

    MSE_test_scores = []
    R2_test_scores = []

    x, y = generateData(N)
    z = frankeFunction(x, y, noise)

    '''
    for degree in range(1, p + 1):
        DesignMatrix = computeDesignMatrix(x, y, degree)


        OLS = Regression(DesignMatrix, z)
        OLS.splitData(0.2)
        OLS.scaleData()
        OLS.fit()
        OLS.predict()
        OLS.predict(test=True)

        MSE_train_scores[degree - 1] = Analysis.MSE(OLS.z_train, OLS.z_tilde))
        R2_train_scores[degree - 1] = Analysis.R2(OLS.z_train, OLS.z_tilde))
        #Betas[:, degree - 1] = OLS.beta

        MSE_test_scores.append(Analysis.MSE(OLS.z_test, OLS.z_predict))
        R2_test_scores.append(Analysis.R2(OLS.z_test, OLS.z_predict))
    '''

    DesignMatrix = computeDesignMatrix(x, y, p)

    Analysis.plot_confidence_intervals(DesignMatrix, z, p)

    '''
    Analysis.plot_mse_vs_complexity(MSE_train_scores, 
                                    MSE_test_scores, 
                                    N, 
                                    noise, 
                                    p, 
                                    "mse_vs_complexity")
    '''

N = int(sys.argv[1])
noisy = float(sys.argv[2])
poly_degree = int(sys.argv[3])
#bootstraps = int(sys.argv[4])

partA(N, noisy, poly_degree)
#partB(N, noisy, poly_degree, bootstraps, False)