import numpy as np

from analysis import Analysis
from tools import computeDesignMatrix
from regression import Regression
from ridge import Ridge
from lasso import Lasso

class Bootstrap(object):


    @staticmethod
    def bootstrap(x, y, z, degree, bootstraps, method, lmbda=0.0):
        """Performs the bootstrap resampling method on given dataset.
        
        Args:
            x, y, z: dataset to be resampled
            degree: degree of polynomial
            bootstraps: number of iterations
            method: OLS v Ridge v Lasso
            lmbda: default value 0.0, if method=Ridge then lmbda will be given
        Returns:
            mse: Mean Squared Error for the test set
            mse_train: Mean Squared Error for the training set
            bias
            variance
            beta_average
            beta_variance
        """

        DesignMatrix = computeDesignMatrix(x, y, degree)

        # create model of type
        if method == 'OLS':
            MODEL = Regression(DesignMatrix, z)
        elif method == 'Ridge':
            MODEL = Ridge(DesignMatrix, z, lmbda)
        elif method == 'Lasso':
            pass

        # split and scale data
        MODEL.splitData(0.2)
        MODEL.scaleData()

        # for saving results
        betas = np.zeros((MODEL.X_train.shape[1], bootstraps))
        z_predicts = np.zeros((MODEL.z_test.shape[0], bootstraps))
        z_tildes = np.zeros((MODEL.z_train.shape[0], bootstraps))
        z_train_samples = np.zeros((MODEL.z_train.shape[0], bootstraps))
        
        samples = MODEL.X_train.shape[0]
        for i in range(bootstraps):

            # get random indices
            random_indices = np.random.randint(0, samples, samples)

            # draw samples
            X_train_boot = MODEL.X_train[random_indices]
            z_train_boot = MODEL.z_train[random_indices]
            
            # fit 
            beta = MODEL.fit(*[X_train_boot, z_train_boot])

            # predict
            z_tilde = X_train_boot @ beta
            z_predict = MODEL.X_test @ beta

            # add result
            betas[:,i] = beta
            z_predicts[:,i] = z_predict
            z_tildes[:,i] = z_tilde
            z_train_samples[:,i] = z_train_boot
        

        # calculate average and variance for all Beta_0, Beta_1, and so on..
        beta_average = np.average(betas, axis=1)
        beta_variance = np.var(betas, axis=1)

        z_test = MODEL.z_test.reshape((len(MODEL.z_test), 1))

        mse = np.mean(np.mean((z_test - z_predicts)**2, axis=1, keepdims=True))
        mse_train = np.mean(np.mean((z_tildes - z_train_samples)**2, axis=1, keepdims=True))
        bias = np.mean((z_test - np.mean(z_predicts, axis=1, keepdims=True))**2)
        variance = np.mean(np.var(z_predicts, axis=1, keepdims=True))

        return mse, mse_train, bias, variance, beta_average, beta_variance

class CrossValidation(object):

    @staticmethod
    def kFoldSplit(folds, N_data):
        """Shuffles indices of given length and divides indices into test-indices and train-indices.
        Returns test-indices and train-indices for each fold.

        Args:
            folds: number of folds
            N_data: number of indices to be split/divided
        Return:
            train_indices: 2D list where each index in list is a list of indices
            test_indices: 2D list where each i in list is a list of indices
        """

        if N_data % folds != 0:
            print("Not able to divide dataset in k = " + str(k) + " folds evenly!")
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
    def kFoldCrossValidation(x, y, z, degree, folds, method, lmbda=0.0):
        """Performs the k-fold cross validation rsampling method.

        Args:
            x, y, z: dataset to be resampled
            degree: degree of polynomial
            bootstraps: number of iterations
            method: OLS v Ridge v Lasso
            lmbda: default value 0.0, if method=Ridge then lmbda will be given
        Returns:
            average_mse_test: average Mean Squared Error for the test set
        """

        DesignMatrix = computeDesignMatrix(x, y, degree)
        
        if method == 'OLS':
            MODEL = Regression(DesignMatrix, z)
        elif method == 'Ridge':
            MODEL = Ridge(DesignMatrix, z, lmbda)
        elif method == 'Lasso':
            pass
        
        MODEL.scaleData()

        MSE_test_kfold = np.zeros(folds)
        MSE_train_kfold = np.zeros(folds)

        train_folds, test_folds = CrossValidation.kFoldSplit(folds, MODEL.X.shape[0])
        
        i = 0
        for train_indices, test_indices in zip(train_folds, test_folds):

            # get test and train folds
            X_train = MODEL.X[train_indices]
            z_train = MODEL.z[train_indices]

            X_test = MODEL.X[test_indices]
            z_test = MODEL.z[test_indices]

            # fit and predict
            beta = MODEL.fit(*[X_train, z_train])
            z_tilde = X_train @ beta
            z_predict = X_test @ beta

            # add results
            MSE_test_kfold[i] = Analysis.MSE(z_test, z_predict)
            MSE_train_kfold[i] = Analysis.MSE(z_tilde, z_train)
            i += 1

        average_mse_test = np.mean(MSE_test_kfold)
        average_mse_train = np.mean(MSE_train_kfold)

        return average_mse_test, average_mse_train