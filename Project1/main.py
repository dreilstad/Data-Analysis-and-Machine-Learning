import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from regression import Regression
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

class Project:

    def __init__(self):
        self.regression = None

    
    def generateData(self, N, noise=False, noise_strength=0.0):

        x1, x2 = np.random.uniform(0, 1, size=(2, N))
        self.x1 = x1
        self.x2 = x2

        if noise:
            addNoise(noise_strength)
        
    def scaleData(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled

    def addNoise(self, noise_strength):
        
        noise_amount = noise_strength * np.random.normal(0, 1, self.N)
        self.x1 += noise_amount
        self.x2 += noise_amount

    def getPolynomial(self, degree=5):
        return polyvander2d(self.x1, self.x2, (degree, degree))
    
    def getFrankeFunction(self, x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2)) 
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) 
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4


def partA(N, degree):

    partA = Project()
    partA.generateData(N)

    y = partA.getFrankeFunction(partA.x1, partA.x2)
    X = partA.getPolynomial()

    ols = Regression(X, y)
    ols.splitData()
    X_train_scaled, X_test_scaled = ols.scaleData()
    
    beta_OLS = ols.beta()
    y_predict = ols.fit()



    print(X)


partA(10, 5)
