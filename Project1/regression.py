import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Regression(object):

    def __init__(self, X, z, lmbda=0, method='invert'):
        
        self.X = X
        self.z = z
        self.lmbda = lmbda
        self.method = method
    
    def beta(self):
        try:
            return self.beta
        except AttributeError:
            self.beta = OrdinaryLeastSquares()
            return self.beta

    def OrdinaryLeastSquares(self):
        return np.linalg.pinv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.z_train


    def fit(self):
        self.beta = self.OrdinaryLeastSquares()
    
    def predict(self, test=False):

        if not test:
            self.z_tilde = self.X_train @ self.beta
            return self.z_tilde
        else:
            self.z_predict = self.X_test @ self.beta
            return self.z_predict


    def SVD(self):
        pass

    def scaleData(self):
        '''
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        '''

        self.X_train -= np.mean(self.X_train)
        self.X_test -= np.mean(self.X_test)

    def splitData(self, testdata_size):
        X_train, X_test, z_train, z_test = train_test_split(self.X, self.z, test_size=testdata_size)
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test
    
    
