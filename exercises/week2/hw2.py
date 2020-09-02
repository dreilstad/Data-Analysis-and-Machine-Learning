import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


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

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def varianceOfCoeff(DesignMatrix, f, f_predict):
    N,p = DesignMatrix.shape
    variance = (1/(N-p-1))*sum((f-f_predict)**2)
    return np.linalg.inv(X_train.T @ X_train) * variance

# Generate dataset
noise_factor = 0.1
x = np.random.rand(100)
y = 2.0+5*x*x+noise_factor*np.random.randn(100)

# The design matrix now as function of a given polynomial 
p = 3
X = np.zeros((len(x),3))
X[:,0] = 1.0
X[:,1] = x
X[:,2] = x**2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# matrix inversion to find beta
OLSbeta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train 
print(OLSbeta)
# variance
print(np.diagonal(np.linalg.inv(X_train.T @ X_train)))
# and then make the prediction
ytildeOLS = X_train @ OLSbeta
print("Training R2 for OLS")
print(R2(y_train,ytildeOLS))
print("Training MSE for OLS")
print(MSE(y_train,ytildeOLS))
ypredictOLS = X_test @ OLSbeta
print("Test R2 for OLS")
print(R2(y_test,ypredictOLS))
print("Test MSE OLS")
print(MSE(y_test,ypredictOLS))

# Repeat now for Ridge regression and various values of the regularization parameter 
I = np.eye(p,p)

# Decide which values of lambda to use
nlambdas = 200
MSEPredict = np.zeros(nlambdas) 
MSETrain = np.zeros(nlambdas)
MSEScikit_train = np.zeros(nlambdas)
MSEScikit_predict = np.zeros(nlambdas)
lambdas = np.logspace(-4, 1, nlambdas) 
for i in range(nlambdas):
    lmb = lambdas[i]

    clf_ridge = skl.Ridge(alpha=lmb).fit(X_train, y_train)
    yridge_test = clf_ridge.predict(X_test)
    yridge_train = clf_ridge.predict(X_train)

    Ridgebeta = np.linalg.inv(X_train.T @ X_train+lmb*I) @ X_train.T @ y_train # and then make the prediction
    
    ytildeRidge = X_train @ Ridgebeta
    ypredictRidge = X_test @ Ridgebeta
    
    MSEPredict[i] = MSE(y_test,ypredictRidge)
    MSETrain[i] = MSE(y_train,ytildeRidge)
    MSEScikit_train[i] = mean_squared_error(y_train, yridge_train)
    MSEScikit_predict[i] = mean_squared_error(y_test, yridge_test)


print(Ridgebeta)
print(np.diagonal(np.linalg.inv(X_train.T @ X_train+lmb*I)))

# Now plot the resulys 
plt.plot(np.log10(lambdas), MSETrain, 'b-', label='MSE Ridge train')
plt.plot(np.log10(lambdas), MSEPredict, 'r-', label='MSE Ridge test')
plt.plot(np.log10(lambdas), MSEScikit_predict, 'g-', label='MSE Ridge scikit test')
plt.plot(np.log10(lambdas), MSEScikit_train, 'y-', label='MSE Ridge scikit train')
plt.xlabel('log10(lambda)') 
plt.ylabel('MSE') 
plt.legend()
save_fig('Scikit-MSE_vs_manual-MSE')
plt.figure()
plt.scatter(y, x)
plt.plot(ytildeRidge)
plt.show()