import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Regression(object):

    def __init__(self, X, y, lmbda=0, method="invert"):
        
        self.X = X
        self.y = y
        self.lmbda = lmbda
        self.method = method
    
    def beta(self):
        try:
            return self.beta
        except AttributeError:
            OrdinaryLeastSquares()
            return self.beta

    def OrdinaryLeastSquares(self):

        if self._method == 'invert':
            self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        else:
            self.beta = SVD()
    
    def fit(self):
        return self.X @ beta()

    def SVD(self):
        pass

    def scaleData(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled

    def splitData(self):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    

X = 0
y = 0
c = 1
b = Regression(X, y)
print(b.predict())