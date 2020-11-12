import numpy as np

class SGD():

    def __init__(self, X, y, beta=None, epochs=100, size_batch=1, method="OLS", lmbda=0.0):
        
        self.X = X
        self.y = y
        self.epochs = epochs

        N,P = X.shape
        self.N = N
        self.P = P

        self.size_batch = size_batch
        self.num_batches = int(N / size_batch)

        if beta is None:
            self.beta = np.random.randn(P, 1).ravel()
        else:
            self.beta = beta
        
        self.method = method

        # ridge
        self.lmbda = lmbda


    def learning_rate_decay(self, t, t0, t1):
        """Calculates the learning rate decay.

        Args:
            t (int): (epoch number) * (number of batches) + (batch number)
            t0 (int): t0 value
            t1 (int): t1 value
        """
        return t0 / (t + t1)

    def gradient(self, batch_indices, beta):
        """Calculating the gradient of the MSE cost function for OLS or Ridge regression.

        Args:
            batch_indices (ndarray): 1D array of indices to use from dataset
            beta (ndarray): 1D array of beta parameters
        """

        X_b = self.X[batch_indices]
        y_b = self.y[batch_indices]

        if self.method == "Ridge":
            return 2 * (X_b.T @ ((X_b @ beta) - y_b) + self.lmbda * beta)
        else:
            return (2/self.size_batch) * X_b.T @ ((X_b @ beta) - y_b)
        

    def fit(self, learning_rate=0.001):
        """Estimates the beta paramaters using standard SGD.
        Shuffles the dataset for each epoch.

        Args:
            learning_rate (float): learning rate, default=0.001
        """

        indices = np.arange(self.N)
        shuffler = np.random.default_rng()
        
        beta = self.beta
        for epoch in range(self.epochs):
        
            shuffler.shuffle(indices)
            for i in range(self.num_batches):
                batch_indices = np.random.choice(indices, self.size_batch, replace=True)
                beta = beta - learning_rate * self.gradient(batch_indices, beta)
        
        return beta.ravel()

    def fit_with_decay(self, t0=5, t1=50):
        """Estimates the beta paramaters using SGD with decaying learning rate.
        Shuffles the dataset for each epoch.

        Args:
            t0 (float): t0 value, default=5
            t0 (float): t0 value, default=50
        """

        indices = np.arange(self.N)
        shuffler = np.random.default_rng()

        beta = self.beta
        for epoch in range(self.epochs):

            shuffler.shuffle(indices)
            for i in range(self.num_batches):
                batch_indices = np.random.choice(indices, self.size_batch, replace=True)
                learning_rate = self.learning_rate_decay(epoch * self.num_batches + i, t0, t1)

                beta = beta - learning_rate * self.gradient(batch_indices, beta)
        
        return beta.ravel()

    def RMSprop(self, learning_rate=0.001, rho=0.9, eps=1e-8):
        """Estimates the beta paramaters using the RMSprop variant of SGD.
        Shuffles the dataset for each epoch.

        Source: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

        Args:
            learning_rate (float): learning rate, default=0.001
            rho (float): rho value, default=0.9
            eps (float): epsilon value, default=1e-8
        """

        indices = np.arange(self.N)
        shuffler = np.random.default_rng()

        gradient_squared = np.zeros(len(self.beta))
        beta = self.beta
        for epoch in range(self.epochs):

            shuffler.shuffle(indices)
            for i in range(self.num_batches):
                batch_indices = np.random.choice(indices, self.size_batch, replace=True)

                gradient = self.gradient(batch_indices, beta)
                gradient_squared = (rho * gradient_squared) + ((1 - rho) * gradient * gradient)

                beta = beta - learning_rate * gradient / np.sqrt(gradient_squared + eps)
        
        return beta.ravel()

    def ADAM(self, learning_rate=0.001, rho_1=0.9, rho_2=0.99, eps=1e-8):
        """Estimates the beta paramaters using the ADAM variant of SGD.
        Shuffles the dataset for each epoch.

        Source: https://arxiv.org/pdf/1412.6980v9.pdf

        Args:
            learning_rate (float): learning rate, default=0.001
            rho_1 (float): rho value, default=0.9
            rho_2 (float): rho value, default=0.99
            eps (float): epsilon value, default=1e-8
        """

        indices = np.arange(self.N)
        shuffler = np.random.default_rng()

        m = np.zeros(len(self.beta))
        s = np.zeros(len(self.beta))
        m_hat = np.zeros(len(self.beta))
        s_hat = np.zeros(len(self.beta))
        
        beta = self.beta
        for epoch in range(self.epochs):

            shuffler.shuffle(indices)
            for i in range(1, self.num_batches + 1):
                batch_indices = np.random.choice(indices, self.size_batch, replace=True)

                gradient = self.gradient(batch_indices, beta)
                m = (rho_1 * m) + ((1 - rho_1) * gradient)
                s = (rho_2 * s) + ((1 - rho_2) * gradient * gradient)

                m_hat = m / (1 - rho_1**i)
                s_hat = s / (1 - rho_2**i)

                beta = beta - learning_rate * (m_hat / (np.sqrt(s_hat) + eps))
        
        return beta.ravel()
