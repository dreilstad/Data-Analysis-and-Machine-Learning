import numpy as np


class LogisticRegression():

    def __init__(self, X, y, onehot, categories, learning_rate, lmbda, epochs, size_batch):

        self.X = X
        self.y = y
        self.onehot = onehot
        self.categories = categories
        self.learning_rate = learning_rate
        self.lmbda = lmbda

        N,P = X.shape
        self.inputs = N
        self.predictors = P
        self.epochs = epochs
        self.size_batch = size_batch

        self.init_beta()

    def init_beta(self):
        """Initializes the beta parameters for each class as a matrix of size (n_predictors, n_categroies)
        """
        self.beta = np.zeros([self.predictors, self.categories])


    def gradient(self, probabilities, batch_indices):
        """Calculates the gradient of cost function and returns the result.

        Args:
            probabilities (ndarray): probabilities for each class on each sample
            batch_indices (ndarray): indices to use form the full dataset X

        Returns:
            gradient of the cost function
        """

        return -(1 / len(batch_indices)) * np.dot(self.X[batch_indices].T, (self.onehot[batch_indices] - probabilities))

    def compute_score_and_probabilities(self, beta, batch_indices):
        """Calculates the scores, feeds the scores into the softmax function to get the probability for each class. 
        Returns the results.

        Args: 
            beta (ndarray): beta parameters for each class
            batch_indices (ndarray): indices to use form the full dataset X

        Returns:
            the probabilites for each class one each sample
        """

        logit_scores = np.dot(self.X[batch_indices], beta)
        probabilities = self.softmax(logit_scores)
        return probabilities

    def fit(self):
        """Estimates the beta parameters for each class using stochastc gradient descent. The dataset is shuffled for each epoch.
        Returns the beta parameters.

        Returns:
            beta parameters for each class
        """

        indices = np.arange(self.inputs)
        shuffler = np.random.default_rng()

        beta = self.beta
        for epoch in range(self.epochs):

            shuffler.shuffle(indices)
            for i in range(1, (self.inputs // self.size_batch) + 1):
                batch_indices = np.random.choice(indices, self.size_batch, replace=True)
            
                probabilities = self.compute_score_and_probabilities(beta, batch_indices)
                gradient = self.gradient(probabilities, batch_indices)

                beta = (1 - self.learning_rate * self.lmbda) * beta - self.learning_rate * gradient
        
        return beta


    def predict(self, X, beta):
        """Predicts the class for each sample of the test set using the beta parameters for each class.
        Returns the prediction.

        Args:
            X (ndarray): test set to predict on
            beta (ndarray): beta parametrs for each class

        Returns:
            prediction for each sample of the test set
            probabilites for each sample on the test set
        """

        logit_scores = np.dot(X, beta)
        probabilities = self.softmax(logit_scores)
        prediction = np.argmax(probabilities, axis=1)
        return prediction, probabilities

    def accuracy(self, target, prediction):
        """Calculates the accuracy score using target values and the predicted values.

        Returns:
            accuracy score
        """
        accuracy = float(sum(prediction == target) / len(target))
        return accuracy

    def softmax(self, z):
        """Applies the softmax function to given argument containing scores.

        Returns:
            probabilities for each class
        """

        exp_term = np.exp(z)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities
