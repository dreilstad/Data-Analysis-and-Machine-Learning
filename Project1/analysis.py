import numpy as np
class Analysis:

    @staticmethod
    def R2(z, z_tilde):
        return 1 - (np.sum((z - z_tilde) ** 2)) / (np.sum((z - np.mean(z)) ** 2))
    
    @staticmethod
    def MSE(z, z_tilde):
        N = np.size(z_tilde)
        return np.sum((z - z_tilde)**2) / N

    @staticmethod
    def betaCoeffVariance(X, z, z_tilde):
        N, p = X.shape
        variance = (1/(N-p-1))*sum((z - z_tilde)**2)
        return np.diagonal(np.linalg.pinv(X.T @ X)) * variance