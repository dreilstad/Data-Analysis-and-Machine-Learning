import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class OrdinaryLeastSquares(object):

    def __init__(self, X, z):
        
        self.X = X
        self.z = z
    
    def beta(self):
        return self.beta

    def fit(self, *args):
        '''Function calculates the beta coefficients using 'inv(X.T @ X) @ X.T @ z',
           the function pinv does the equivalent. Default is using the training set, 
           but arguments can be provided manually.

        Args:
            args (list): a list of a desgin matrix X and a corresponding vector z in order
        Returns:
            a vector with the beta coefficients
        '''
        if len(args) == 0:
            self.beta = np.linalg.pinv(self.X_train) @ self.z_train
            return self.beta
        else:
            self.beta = np.linalg.pinv(args[0]) @ args[1]
            return self.beta

    def predict(self, test=False):
        '''Function predicts using the design matrix and beta vector.
           Checks if the prediction is on the test set

        Args:
            test (bool): if the prediction is done on the training set or the test set
        Returns:
            vector of size (Nx1) with the predicted values
        '''

        if not test:
            self.z_tilde = self.X_train @ self.beta
            return self.z_tilde
        else:
            self.z_predict = self.X_test @ self.beta
            return self.z_predict

    def scaleData(self):
        '''Function scales the data by using StandardScaler from Scikit-Learn.
        Before scaling, the intercept is removed and afterwards added back to the design matrix

        If the dataset has not been split, the scaling is done manually for the whole dataset.
        '''
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
        '''Function splits the dataset into a training set and a test set with a given percentage split as argument

        Args:
            testdata_size (float): usually has a value of 0.2
        '''

        X_train, X_test, z_train, z_test = train_test_split(self.X, self.z, test_size=testdata_size)
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test


    def beta_coeff_variance(self):
        '''Function calculates the variance of the beta coefficients. First the variance is calculated and multiplied to the diagonal values of inverse(X.T @ X).

        Returns: 
            a list of the diagonal values of inverse(X.T @ X) multiplied with the variance of the model
        '''

        N, p = self.X_test.shape
        variance = (1/(N-p-1))*sum((self.z_test - self.z_predict)**2)

        beta_coeff_variance = np.diagonal(np.linalg.pinv(self.X_test.T @ self.X_test)) * variance
        return beta_coeff_variance
    
