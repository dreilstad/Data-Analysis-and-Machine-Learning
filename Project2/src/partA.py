import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import analysis as an
import tools as tools

from sgd import SGD
from ols import OrdinaryLeastSquares
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def find_best_learning_schedule(t1_values, N, noise, degrees, epochs=100, size_batch=10):
    """
    Calculates the MSE and plots the MSE as a function of different t1 values using SGD with decay.
    Plots the result.
    """

    for degree in degrees:
        
        x, y = tools.generateData(N)
        X = tools.computeDesignMatrix(x, y, degree)
        z = tools.frankeFunction(x, y, noise=noise)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        X_train, X_test = tools.scale(X_train, X_test)

        mse_test_scores = np.zeros(len(t1_values))

        for i, t1 in enumerate(t1_values):
            print("Finding best learning schedule params: t1=" + str(t1), end="      \r")

            MODEL = SGD(X, y, epochs=epochs, size_batch=size_batch)
            beta = MODEL.fit_with_decay(t0=1, t1=t1)
            z_predict = X_test @ beta

            mse_score = an.MSE(z_test, z_predict)

            mse_test_scores[i] = mse_score


        an.plot_learning_decay(mse_test_scores, t1_values, degree, N, noise, "decay_fit_learning_decay_t0=" + str(1) + "Epochs=" + str(epochs))


def compare_variants_lr(N, noise, degrees, learning_rates, epoch=100, t_0=5, t_1=50, measurement="MSE"):
    """
    Calculates the error using different SGD variants, and plots the error as a function of the learning rate.
    """

    for i, degree in enumerate(degrees):

        x, y = tools.generateData(N)
        X = tools.computeDesignMatrix(x, y, degree)
        z = tools.frankeFunction(x, y, noise=noise)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        X_train, X_test = tools.scale(X_train, X_test)

        MODEL_OLS = OrdinaryLeastSquares(X_train, z_train)
        beta_ols = MODEL_OLS.fit(*[X_train, z_train])
        z_predict_ols = X_test @ beta_ols
        if measurement == "R2":
            score_ols = an.R2(z_test, z_predict_ols)
        elif measurement == "MSE":
            score_ols = an.MSE(z_test, z_predict_ols)

        test_scores = np.zeros((5, len(learning_rates)))

        common_start_beta = np.random.randn(X_train.shape[1], 1).ravel()

        for j, learning_rate in enumerate(learning_rates):
            print("Compare variants: degree=" + str(degree) + ", learning_rate=" + str(learning_rate), end="      \r")
            
            MODEL_STD = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch)
            MODEL_DECAY = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch)
            MODEL_RMS = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch)
            MODEL_ADAM = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch)

            beta_std = MODEL_STD.fit(learning_rate=learning_rate)
            beta_decay = MODEL_DECAY.fit_with_decay(t0=t_0, t1=t_1)
            beta_RMS = MODEL_RMS.RMSprop(learning_rate=learning_rate)
            beta_ADAM = MODEL_ADAM.ADAM(learning_rate=learning_rate)

            z_predict_std = X_test @ beta_std
            z_predict_decay = X_test @ beta_decay
            z_predict_RMS = X_test @ beta_RMS
            z_predict_ADAM = X_test @ beta_ADAM

            if measurement == "R2":
                score_std = an.R2(z_test, z_predict_std)
                score_decay = an.R2(z_test, z_predict_decay)
                score_RMS = an.R2(z_test, z_predict_RMS)
                score_ADAM = an.R2(z_test, z_predict_ADAM)
            elif measurement == "MSE":
                score_std = an.MSE(z_test, z_predict_std)
                score_decay = an.MSE(z_test, z_predict_decay)
                score_RMS = an.MSE(z_test, z_predict_RMS)
                score_ADAM = an.MSE(z_test, z_predict_ADAM)

            test_scores[0][j] = score_std
            test_scores[1][j] = score_decay
            test_scores[2][j] = score_RMS
            test_scores[3][j] = score_ADAM

            prev_epoch = epoch

            prev_beta_std = beta_std
            prev_beta_decay = beta_decay
            prev_beta_RMS = beta_RMS
            prev_beta_ADAM = beta_ADAM

        test_scores[4] = np.full(len(learning_rates), score_ols)

        an.plot_compare_variants_lr(test_scores, 
                                    learning_rates, 
                                    degree, 
                                    N, 
                                    noise, 
                                    "compare_variants_learning_rates_Epochs=" + str(epoch) + "_t0=" + str(t_0) + "_t1=" + str(t_1))
        

def compare_variants_epochs(N, noise, degrees, epochs, learning_rate=0.001, size_batch=10, t_0=5, t_1=50, measurement="MSE"):
    """
    Calculates the error using different SGD variants, and plots the error as a function of the number of epochs.
    """

    for i, degree in enumerate(degrees):

        x, y = tools.generateData(N)
        X = tools.computeDesignMatrix(x, y, degree)
        z = tools.frankeFunction(x, y, noise=noise)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        X_train, X_test = tools.scale(X_train, X_test)

        MODEL_OLS = OrdinaryLeastSquares(X_train, z_train)
        beta_ols = MODEL_OLS.fit(*[X_train, z_train])
        z_predict_ols = X_test @ beta_ols
        if measurement == "R2":
            score_ols = an.R2(z_test, z_predict_ols)
        elif measurement == "MSE":
            score_ols = an.MSE(z_test, z_predict_ols)

        test_scores = np.zeros((5, len(epochs)))

        prev_beta_std = None
        prev_beta_decay = None
        prev_beta_RMS = None
        prev_beta_ADAM = None

        prev_epoch = 0

        for j, epoch in enumerate(epochs):
            print("Compare variants: degree=" + str(degree) + ", epochs=" + str(epoch))
            
            MODEL_STD = SGD(X_train, z_train, beta=prev_beta_std, epochs=epoch - prev_epoch, size_batch=size_batch)
            MODEL_DECAY = SGD(X_train, z_train, beta=prev_beta_decay, epochs=epoch - prev_epoch, size_batch=size_batch)
            MODEL_RMS = SGD(X_train, z_train, beta=prev_beta_RMS, epochs=epoch - prev_epoch, size_batch=size_batch)
            MODEL_ADAM = SGD(X_train, z_train, beta=prev_beta_ADAM, epochs=epoch - prev_epoch, size_batch=size_batch)

            beta_std = MODEL_STD.fit(learning_rate=learning_rate)
            beta_decay = MODEL_DECAY.fit_with_decay(t0=t_0, t1=t_1)
            beta_RMS = MODEL_RMS.RMSprop(learning_rate=learning_rate)
            beta_ADAM = MODEL_ADAM.ADAM(learning_rate=learning_rate)

            z_predict_std = X_test @ beta_std
            z_predict_decay = X_test @ beta_decay
            z_predict_RMS = X_test @ beta_RMS
            z_predict_ADAM = X_test @ beta_ADAM

            if measurement == "R2":
                score_std = an.R2(z_test, z_predict_std)
                score_decay = an.R2(z_test, z_predict_decay)
                score_RMS = an.R2(z_test, z_predict_RMS)
                score_ADAM = an.R2(z_test, z_predict_ADAM)
            elif measurement == "MSE":
                score_std = an.MSE(z_test, z_predict_std)
                score_decay = an.MSE(z_test, z_predict_decay)
                score_RMS = an.MSE(z_test, z_predict_RMS)
                score_ADAM = an.MSE(z_test, z_predict_ADAM)

            test_scores[0][j] = score_std
            test_scores[1][j] = score_decay
            test_scores[2][j] = score_RMS
            test_scores[3][j] = score_ADAM

            prev_epoch = epoch

            prev_beta_std = beta_std
            prev_beta_decay = beta_decay
            prev_beta_RMS = beta_RMS
            prev_beta_ADAM = beta_ADAM

        test_scores[4] = np.full(len(epochs), score_ols)

        an.plot_compare_variants_epoch(test_scores, 
                                       epochs, 
                                       degree, 
                                       N, 
                                       noise, 
                                       "compare_variants_epochs_LR=" + str(learning_rate) + "_t0=" + str(t_0) + "_t1=" + str(t_1))
        

def compare_variants_mb(N, noise, degrees, mini_batches, learning_rate=0.01, epoch=100, t_0=5, t_1=50, measurement="MSE"):
    """
    Calculates the error using different SGD variants, and plots the error as a function of the size of the minibatches.
    """

    for i, degree in enumerate(degrees):

        x, y = tools.generateData(N)
        X = tools.computeDesignMatrix(x, y, degree)
        z = tools.frankeFunction(x, y, noise=noise)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        X_train, X_test = tools.scale(X_train, X_test)

        MODEL_OLS = OrdinaryLeastSquares(X_train, z_train)
        beta_ols = MODEL_OLS.fit(*[X_train, z_train])
        z_predict_ols = X_test @ beta_ols
        if measurement == "R2":
            score_ols = an.R2(z_test, z_predict_ols)
        elif measurement == "MSE":
            score_ols = an.MSE(z_test, z_predict_ols)

        test_scores = np.zeros((5, len(mini_batches)))

        common_start_beta = np.random.randn(X_train.shape[1], 1).ravel()

        for j, size_batch in enumerate(mini_batches):
            print("Compare variants: degree=" + str(degree) + ", mini_batches=" + str(N//size_batch))
            
            MODEL_STD = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch)
            MODEL_DECAY = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch)
            MODEL_RMS = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch)
            MODEL_ADAM = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch)

            beta_std = MODEL_STD.fit(learning_rate=learning_rate)
            beta_decay = MODEL_DECAY.fit_with_decay(t0=t_0, t1=t_1)
            beta_RMS = MODEL_RMS.RMSprop(learning_rate=learning_rate)
            beta_ADAM = MODEL_ADAM.ADAM(learning_rate=learning_rate)

            z_predict_std = X_test @ beta_std
            z_predict_decay = X_test @ beta_decay
            z_predict_RMS = X_test @ beta_RMS
            z_predict_ADAM = X_test @ beta_ADAM

            if measurement == "R2":
                score_std = an.R2(z_test, z_predict_std)
                score_decay = an.R2(z_test, z_predict_decay)
                score_RMS = an.R2(z_test, z_predict_RMS)
                score_ADAM = an.R2(z_test, z_predict_ADAM)
            elif measurement == "MSE":
                score_std = an.MSE(z_test, z_predict_std)
                score_decay = an.MSE(z_test, z_predict_decay)
                score_RMS = an.MSE(z_test, z_predict_RMS)
                score_ADAM = an.MSE(z_test, z_predict_ADAM)

            test_scores[0][j] = score_std
            test_scores[1][j] = score_decay
            test_scores[2][j] = score_RMS
            test_scores[3][j] = score_ADAM

            prev_epoch = epoch

            prev_beta_std = beta_std
            prev_beta_decay = beta_decay
            prev_beta_RMS = beta_RMS
            prev_beta_ADAM = beta_ADAM

        test_scores[4] = np.full(len(mini_batches), score_ols)

        an.plot_compare_variants_mb(test_scores, 
                                    mini_batches, 
                                    degree, 
                                    N, 
                                    noise, 
                                    "compare_variants_minibatches_LR=" + str(learning_rate) + "_Epochs=" + str(epoch) + "_t0=" + str(t_0) + "_t1=" + str(t_1))


def compare_variant_lr_lmbda(N, noise, degrees, learning_rates, lmbdas, epoch=100, size_batch=10, t0=5, t1=50, measurement="R2"):
    """
    Calculates the error using different SGD variants, and plots the error as a function of the learning rates and 
    regularization parameter in a heatmap.
    """

    for degree in degrees:

        x, y = tools.generateData(N)
        X = tools.computeDesignMatrix(x, y, degree)
        z = tools.frankeFunction(x, y, noise=noise)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        X_train, X_test = tools.scale(X_train, X_test)

        
        test_scores = np.zeros((4, len(learning_rates), len(lmbdas)))
        for i, learning_rate in enumerate(learning_rates):

            common_start_beta = np.random.randn(X_train.shape[1], 1).ravel()
            for j, lmbda in enumerate(lmbdas):
                print("LR vs Lambda: degree=" + str(degree) + ", LR=" + str(learning_rate) + ", Lambda=" + str(lmbda))

                MODEL_STD = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch, method="Ridge", lmbda=lmbda)
                MODEL_DECAY = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch, method="Ridge", lmbda=lmbda)
                MODEL_RMS = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch, method="Ridge", lmbda=lmbda)
                MODEL_ADAM = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch, method="Ridge", lmbda=lmbda)

                beta_std = MODEL_STD.fit(learning_rate=learning_rate)
                beta_decay = MODEL_DECAY.fit_with_decay(t0=t0, t1=t1)
                beta_RMS = MODEL_RMS.RMSprop(learning_rate=learning_rate)
                beta_ADAM = MODEL_ADAM.ADAM(learning_rate=learning_rate)

                z_predict_std = X_test @ beta_std
                z_predict_decay = X_test @ beta_decay
                z_predict_RMS = X_test @ beta_RMS
                z_predict_ADAM = X_test @ beta_ADAM

                if measurement == "R2":
                    score_std = an.R2(z_test, z_predict_std)
                    score_decay = an.R2(z_test, z_predict_decay)
                    score_RMS = an.R2(z_test, z_predict_RMS)
                    score_ADAM = an.R2(z_test, z_predict_ADAM)
                elif measurement == "MSE":
                    score_std = an.MSE(z_test, z_predict_std)
                    score_decay = an.MSE(z_test, z_predict_decay)
                    score_RMS = an.MSE(z_test, z_predict_RMS)
                    score_ADAM = an.MSE(z_test, z_predict_ADAM)
                
                test_scores[0][i][j] = score_std
                test_scores[1][i][j] = score_decay
                test_scores[2][i][j] = score_RMS
                test_scores[3][i][j] = score_ADAM
        

        an.plot_lmbda_vs_lr_heatmap(test_scores[0], 
                                    learning_rates, 
                                    lmbdas, 
                                    degree, 
                                    "Standard", 
                                    N, 
                                    noise, 
                                    measurement,
                                    "lmbda_vs_lr_heatmap_Epochs=" + str(epoch) + "_size_batch" + str(size_batch))

        an.plot_lmbda_vs_lr_heatmap(test_scores[1], 
                                    learning_rates, 
                                    lmbdas, 
                                    degree, 
                                    "Decay", 
                                    N, 
                                    noise, 
                                    measurement,
                                    "lmbda_vs_lr_heatmap_Epochs=" + str(epoch) + "_size_batch" + str(size_batch) + "_t0=" + str(t0) + "_t1=" + str(t1))

        an.plot_lmbda_vs_lr_heatmap(test_scores[2], 
                                    learning_rates, 
                                    lmbdas, 
                                    degree, 
                                    "RMSprop", 
                                    N, 
                                    noise,
                                    measurement,
                                    "lmbda_vs_lr_heatmap_Epochs=" + str(epoch) + "_size_batch" + str(size_batch))

        an.plot_lmbda_vs_lr_heatmap(test_scores[3], 
                                    learning_rates, 
                                    lmbdas, 
                                    degree, 
                                    "ADAM", 
                                    N, 
                                    noise, 
                                    measurement,
                                    "lmbda_vs_lr_heatmap_Epochs=" + str(epoch) + "_size_batch" + str(size_batch))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Project 2 - Part A - How to use script', add_help=False)
    parser._action_groups.pop()
    possible_args = parser.add_argument_group('possible arguments')

    possible_args.add_argument('-N', '--N_datapoints', 
                               type=int, 
                               required=True,
                               help='Specify number of datapoints')

    possible_args.add_argument('-no', '--noise', 
                               type=float, 
                               required=False,
                               default=0.0,
                               help='Specify noise to add')

    possible_args.add_argument('-d', '--degree', 
                               type=int, 
                               required=True,
                               help='Specify polynomial degree')

    possible_args.add_argument('-t', '--type', 
                               type=str, 
                               required=True,
                               choices=['epochs', 'lr', 'mb', 't1', 'heatmap', 'decay'],
                               help='Choose parameter to compare')

    possible_args.add_argument('-e', '--epochs', 
                               type=int, 
                               required=False,
                               default=100,
                               help='Specify number of epochs')

    possible_args.add_argument('-lr', '--learning_rate', 
                               type=float, 
                               required=False,
                               default=0.01,
                               help='Specify learning rate')   
    
    possible_args.add_argument('-h', '--help',
                               action='help',
                               help='Helpful message showing flags and usage of code for part A')

    args = parser.parse_args()

    N = args.N_datapoints
    noise = args.noise
    poly_degrees = args.degree
    params = args.type
    epochs = args.epochs
    learning_rate = args.learning_rate

    all_hyperparams = {"epochs":np.arange(10, 3001, 10),
                       "lr":[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                       "mb":[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
                       "t1":[10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]}

    lmbdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    if params == "t1":
        find_best_learning_schedule(all_hyperparams[params], N, noise, [poly_degrees], epochs=epochs)
    elif params == "lr":
        compare_variants_lr(N, noise, [poly_degrees], all_hyperparams[params], epoch=epochs)
    elif params == "epochs":
        compare_variants_epochs(N, noise, [poly_degrees], all_hyperparams[params], learning_rate=learning_rate)
    elif params == "mb":
        compare_variants_mb(N, noise, [poly_degrees], all_hyperparams[params], epoch=epochs, learning_rate=learning_rate)
    elif params == "heatmap":
        compare_variant_lr_lmbda(N, noise, [poly_degrees], all_hyperparams["lr"], lmbdas, epoch=epochs, measurement="MSE")
