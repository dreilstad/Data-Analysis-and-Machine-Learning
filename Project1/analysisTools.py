import numpy as np
class Analysis:

    @staticmethod
    def R2(y, ytilde):
        return 1 - np.sum((y - ytilde) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    @staticmethod
    def MSE(y, ytilde):
        n = np.size(ytilde)
        return np.sum((y-ytilde)**2)/n
    
    @staticmethod
    def varianceOfCoeff(DesignMatrix, f, f_predict):
        N,p = DesignMatrix.shape
        variance = (1/(N-p-1))*sum((f-f_predict)**2)
        #return np.linalg.inv(X_train.T @ X_train) * variance