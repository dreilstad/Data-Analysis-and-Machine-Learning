import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Ridge(object):

    def __init__(self, X, z, lmbda):
        
        self.X = X
        self.z = z
        self.lmbda = lmbda
    
    def beta(self):
        return self.beta


    def fit(self, *args):

        if len(args) == 0:
            A = (self.X_train.T @ self.X_train) + np.eye(self.X_train.shape[1]) * self.lmbda
            self.beta = self.SVD(A) @ self.X_train.T @ self.z_train
            return self.beta
        else:
            X = args[0]
            z = args[1]
            A = (X.T @ X) + np.eye(X.shape[1]) * self.lmbda
            self.beta = self.SVD(A) @ X.T @ z
            return self.beta
    
    def predict(self, test=False):
        if not test:
            self.z_tilde = self.X_train @ self.beta
            return self.z_tilde
        else:
            self.z_predict = self.X_test @ self.beta
            return self.z_predict


    def SVD(self, X):
        
        # decomposition
        U, s, VT = np.linalg.svd(X)
        invD = np.zeros((len(U),len(VT)))

        for i in range(0,len(VT)):
            invD[i,i]=1/s[i]

        UT = np.transpose(U)
        V = np.transpose(VT)

        return np.matmul(V,np.matmul(invD,UT))

    def scaleData(self):
        """
        Scales the data by subtracting the mean. 
        Before scaling, the intercept is removed and afterwards added back to the design matrix
        """
        try:
            self.X_train = self.X_train[:,1:]
            self.X_test = self.X_test[:,1:]

            scaler = StandardScaler(with_mean=True, with_std=False)
            scaler.fit(self.X_train)
            self.X_train = scaler.transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)


            self.X_train = np.c_[np.ones(self.X_train.shape[0]), self.X_train]
            self.X_test = np.c_[np.ones(self.X_test.shape[0]), self.X_test]

        except AttributeError:

            self.X = self.X[:,1:]
            self.X -= np.mean(self.X)
            self.X = np.c_[np.ones(self.X.shape[0]), self.X]

    def splitData(self, testdata_size):
        X_train, X_test, z_train, z_test = train_test_split(self.X, self.z, test_size=testdata_size)
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

    def beta_coeff_variance(self):
        N, p = self.X_test.shape
        variance = (1/(N-p-1))*sum((self.z_test - self.z_predict)**2)

        beta_coeff_variance = np.diagonal(self.SVD(self.X_test.T @ self.X_test)) * variance
        return beta_coeff_variance
    
    
