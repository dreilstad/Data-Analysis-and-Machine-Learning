import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from regression import Regression
from analysis import Analysis
from franke import plotFranke
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

np.random.seed(2020)

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

def frankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) 
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) 
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

'''
def bootstrap(N, DesignMatrix, z, k):

    for i in range(k):

        randomIndices = np.random.randint(0, N, N)
        Xi = DesignMatrix[randomIndices,:]

        OLS_i = Regression(DesignMatrix, z)
        OLS_i.splitData(0.2)
        OLS_i.fit()
        OLS_i.predict()
'''


def partA(N, noise):

    MSE_training_scores = []
    R2_training_scores = []
    Betas_training = []

    MSE_test_scores = []
    R2_test_scores = []
    Betas_test = []

    '''
    MSE_scores_noisy = []
    R2_scores_noisy = []
    Betas_noisy = []
    '''

    for degree in range(2, 6):
        
        x, y = generateData(N)
        #x_noisy, y_noisy = addNoise(x.copy(), y.copy(), noise)
        
        z = frankeFunction(x, y)
        #z_noisy = frankeFunction(x_noisy, y_noisy)

        DesignMatrix = computeDesignMatrix(x, y, degree)
        #DesignMatrix_noisy = computeDesignMatrix(x_noisy, y_noisy, degree)

        OLS = Regression(DesignMatrix, z)
        OLS.splitData(0.2)
        OLS.scaleData()
        OLS.fit()
        OLS.predict()
        OLS.predict(test=True)

        MSE_training_scores.append(Analysis.MSE(OLS.z_train, OLS.z_tilde))
        R2_training_scores.append(Analysis.R2(OLS.z_train, OLS.z_tilde))
        #Betas_training.append(Analysis.betaCoeffVariance(OLS.X_train, OLS.z_train, OLS.z_tilde))

        MSE_test_scores.append(Analysis.MSE(OLS.z_test, OLS.z_predict))
        R2_test_scores.append(Analysis.R2(OLS.z_test, OLS.z_predict))
        
        '''
        OLS_noisy = Regression(DesignMatrix_noisy, z_noisy)
        OLS_noisy.splitData(0.2)
        OLS_noisy.scaleData()

        Beta_OLS_noisy = OLS_noisy.beta()
        z_tilde_OLS_noisy = OLS_noisy.fit()

        MSE_scores_noisy.append(Analysis.MSE(OLS_noisy))
        R2_scores_noisy.append(Analysis.R2(OLS_noisy))
        Betas_noisy.append(Analysis.betaCoeffVariance(OLS_noisy))
        '''

    print("----     Training       ----\n")
    degree = 2
    for i in range(len(MSE_training_scores)):
        print("Degree " + str(degree))
        print("MSE: " + str(MSE_training_scores[i]))
        print("R2: " + str(R2_training_scores[i]))
        print("")
        degree += 1

    print("----     Test       ----\n")
    degree = 2
    for i in range(len(MSE_test_scores)):
        print("Degree " + str(degree))
        print("MSE: " + str(MSE_test_scores[i]))
        print("R2: " + str(R2_test_scores[i]))
        print("")
        degree += 1

N = 500
partA(N, 0.0)