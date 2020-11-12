import numpy as np
import analysis as an

from neural_network import FFNN
from ols import OrdinaryLeastSquares
from ridge import Ridge
from sklearn.neural_network import MLPRegressor

class CrossValidation(object):

    @staticmethod
    def kFoldSplit(folds, N_data):
        '''Shuffles indices of given length and divides indices into test-indices and train-indices.
        Returns test-indices and train-indices for each fold.

        Args:
            folds (int): number of folds
            N_data (int): number of indices to be split/divided
        Return:
            train_indices (ndarray): 2D list where each index in list is a list of indices
            test_indices (ndarray): 2D list where each i in list is a list of indices
        '''

        if N_data % folds != 0:
            print("Not able to divide dataset in k = " + str(folds) + " folds evenly!")
            exit(1)

        len_fold = N_data//folds

        # shuffles dataset
        random_indices = np.arange(N_data)
        shuffler = np.random.default_rng()
        shuffler.shuffle(random_indices)

        train_indices = []
        test_indices = []

        for k in range(folds):

            # calculate the start and end index of testfold with current k
            test_start_index = k * len_fold
            test_end_index = len_fold * (k + 1)

            # slices by start and end index of testfold with current k, 
            # concatenates the remaining indices which becomes training set
            test_indices.append(random_indices[test_start_index:test_end_index])
            train_indices.append(np.hstack((random_indices[:test_start_index], random_indices[test_end_index:])))

            # shuffles dataset for each k
            shuffler.shuffle(random_indices)
            
        return train_indices, test_indices

    @staticmethod
    def kFoldCrossValidation(model, X, z, folds, lmbda=0.0, learning_rate=0.001, epochs=500, size_batch=10):
        """Performs the k-fold cross validation resampling method.

        The dataset is divided into folds and each fold is the test fold once.
        Example for k = 4:
        k=1: |test|train|train|train|

        k=2: |train|test|train|train|

        k=3: |train|train|test|train|

        k=4: |train|train|train|test|

        Args:
            x, y, z (ndarray): dataset to be resampled
            degree (int): degree of polynomial
            folds (int): number of folds
            method (string): OLS v Ridge v Lasso
            lmbda (float): lambda value
        Returns:
            average_mse_test (ndarray): average msefor the test set
            average_mse_train (ndarray): average mse for the train set
            average_r2_test (ndarray): average r2 for the test set
        """

        R2_test_kfold = np.zeros(folds)
        MSE_test_kfold = np.zeros(folds)
        MSE_train_kfold = np.zeros(folds)

        train_folds, test_folds = CrossValidation.kFoldSplit(folds, X.shape[0])
        
        i = 0
        for train_indices, test_indices in zip(train_folds, test_folds):

            # get test and train folds
            X_train = X[train_indices]
            z_train = z[train_indices]

            X_test = X[test_indices]
            z_test = z[test_indices]

            # fit and predict
            if isinstance(model, (OrdinaryLeastSquares, Ridge)):
                beta = model.fit(X_train, z_train)
                z_tilde = X_train @ beta
                z_predict = X_test @ beta
            elif isinstance(model, FFNN):
                model.train(X_train, z_train, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
                z_tilde = model.predict(X_train)
                z_predict = model.predict(X_test)
            elif isinstance(model, MLPRegressor):
                model.fit(X_train, z_train)
                z_tilde = model.predict(X_train)
                z_predict = model.predict(X_test)


            # add results
            R2_test_kfold[i] = an.R2(z_test, z_predict)
            MSE_test_kfold[i] = an.MSE(z_test, z_predict)
            MSE_train_kfold[i] = an.MSE(z_tilde, z_train)
            i += 1

        average_mse_test = np.mean(MSE_test_kfold)
        average_mse_train = np.mean(MSE_train_kfold)
        average_r2_test = np.mean(R2_test_kfold)


        return average_mse_test, average_mse_train, average_r2_test