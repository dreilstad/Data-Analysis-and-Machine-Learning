import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import analysis as an
import tools as tools
from sgd import SGD
from ols import OrdinaryLeastSquares


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def find_best_learning_schedule(t0_values, t1_values, N, noise, degree, epochs=100, measurement="R2"):


    mse_test_scores = np.zeros((len(t0_values), len(t1_values)))
    r2_test_scores = np.zeros((len(t0_values), len(t1_values)))


    x, y = tools.generateData(N)
    X = tools.computeDesignMatrix(x, y, degree)
    z = tools.frankeFunction(x, y, noise=noise)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    X_train, X_test = tools.scale(X_train, X_test)

    for i, t0 in enumerate(t0_values):
        for j, t1 in enumerate(t1_values):
            print("Finding best learning schedule params: t0=" + str(t0) + ", t1=" + str(t1), end="      \r")

            MODEL = SGD(X, y, epochs=epochs)
            beta = MODEL.fit_with_decay(t0=t0, t1=t1)
            z_predict = X_test @ beta

            mse_score = an.MSE(z_test, z_predict)
            r2_score = an.R2(z_test, z_predict)

            mse_test_scores[i][j] = mse_score
            r2_test_scores[i][j] = r2_score

    labels = {r"$t_{0}$ value":t0_values, r"$t_{1}$ value":t1_values}
    an.plot_heatmap(mse_test_scores, 
                    labels, 
                    "Decay", 
                    N, 
                    noise, 
                    1,
                    0.0,
                    "heatmap_mse_learning_schedule_Epochs=" + str(epochs) + "_Degree=" + str(degree))

    an.plot_heatmap(r2_test_scores, 
                    labels, 
                    "Decay", 
                    N, 
                    noise, 
                    0.0,
                    1,
                    "heatmap_r2_learning_schedule_Epochs=" + str(epochs) + "_Degree=" + str(degree))


def computeHeatmapEpochs(N, noise, degrees, epochs, variant, measurement="R2"):

    scores = np.zeros((len(degrees), len(epochs)))

    for i, degree in enumerate(degrees):
        for j, epoch in enumerate(epochs):
            print(variant + ": degree=" + str(degree) + ", epochs=" + str(epoch), end="        \r")
            
            x, y = tools.generateData(N)
            X = tools.computeDesignMatrix(x, y, degree)
            z = tools.frankeFunction(x, y, noise=noise)

            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
            X_train, X_test = tools.scale(X_train, X_test)

            MODEL = SGD(X_train, z_train, epochs=epoch)
        
            if variant == "Standard":
                beta = MODEL.fit()
        
            elif variant == "Decay":
                beta = MODEL.fit_with_decay()
            
            elif variant == "RMSprop":
                beta = MODEL.RMSprop()

            elif variant == "ADAM":
                beta = MODEL.ADAM()

            z_predict = X_test @ beta

            if measurement == "R2":
                score = an.R2(z_test, z_predict)
            elif measurement == "MSE":
                score = an.MSE(z_test, z_predict)

            scores[i][j] = score

    an.plot_heatmap_epochs(scores, 
                           epochs, 
                           degrees, 
                           variant, 
                           N, 
                           noise, 
                           "heatmap_epochs")

                           
def compare_variants_lr(N, noise, degrees, learning_rates, epoch=100, t_0=5, t_1=50, measurement="MSE"):

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
        

def compare_variants_epochs(N, noise, degrees, epochs, learning_rate=0.001, t_0=5, t_1=50, measurement="MSE"):

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
            print("Compare variants: degree=" + str(degree) + ", epochs=" + str(epoch), end="      \r")
            
            MODEL_STD = SGD(X_train, z_train, beta=prev_beta_std, epochs=epoch - prev_epoch)
            MODEL_DECAY = SGD(X_train, z_train, beta=prev_beta_decay, epochs=epoch - prev_epoch)
            MODEL_RMS = SGD(X_train, z_train, beta=prev_beta_RMS, epochs=epoch - prev_epoch)
            MODEL_ADAM = SGD(X_train, z_train, beta=prev_beta_ADAM, epochs=epoch - prev_epoch)

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
        

def compare_variants_mb(N, noise, degrees, mini_batches, learning_rate=0.001, epoch=100, t_0=5, t_1=50, measurement="MSE"):

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
            print("Compare variants: degree=" + str(degree) + ", mini_batches=" + str(N/size_batch), end="      \r")
            
            MODEL_STD = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch)
            MODEL_DECAY = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch)
            MODEL_RMS = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch)
            MODEL_ADAM = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch)

            beta_std = MODEL_STD.fit()
            beta_decay = MODEL_DECAY.fit_with_decay(t0=t_0, t1=t_1)
            beta_RMS = MODEL_RMS.RMSprop()
            beta_ADAM = MODEL_ADAM.ADAM()

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
                print("LR vs Lambda: degree=" + str(degree) + ", LR=" + str(learning_rate) + ", Lambda=" + str(lmbda), end="                        \r")

                MODEL_STD = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch, method="Ridge", lmbda=lmbda)
                MODEL_DECAY = SGD(X_train, n, beta=common_start_beta, epochs=epoch, size_batch=size_batch, method="Ridge", lmbda=lmbda)
                MODEL_RMS = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch, method="Ridge", lmbda=lmbda)
                MODEL_ADAM = SGD(X_train, z_train, beta=common_start_beta, epochs=epoch, size_batch=size_batch, method="Ridge", lmbda=lmbda)

                beta_std = MODEL_STD.fit()
                beta_decay = MODEL_DECAY.fit_with_decay(t0=t0, t1=t1)
                beta_RMS = MODEL_RMS.RMSprop()
                beta_ADAM = MODEL_ADAM.ADAM()

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
                                    "lmbda_vs_lr_heatmap_Epochs=" + str(epoch) + "_size_batch" + str(size_batch))

        an.plot_lmbda_vs_lr_heatmap(test_scores[1], 
                                    learning_rates, 
                                    lmbdas, 
                                    degree, 
                                    "Decay", 
                                    N, 
                                    noise, 
                                    "lmbda_vs_lr_heatmap_Epochs=" + str(epoch) + "_size_batch" + str(size_batch) + "_t0=" + str(t0) + "_t1=" + str(t1))

        an.plot_lmbda_vs_lr_heatmap(test_scores[2], 
                                    learning_rates, 
                                    lmbdas, 
                                    degree, 
                                    "RMSprop", 
                                    N, 
                                    noise, 
                                    "lmbda_vs_lr_heatmap_Epochs=" + str(epoch) + "_size_batch" + str(size_batch))

        an.plot_lmbda_vs_lr_heatmap(test_scores[3], 
                                    learning_rates, 
                                    lmbdas, 
                                    degree, 
                                    "ADAM", 
                                    N, 
                                    noise, 
                                    "lmbda_vs_lr_heatmap_Epochs=" + str(epoch) + "_size_batch" + str(size_batch))

if __name__ == "__main__":
    
    N = int(sys.argv[1])
    noise = float(sys.argv[2])
    poly_degrees = int(sys.argv[3])
    params = sys.argv[4]
    epochs = int(sys.argv[5])

    all_hyperparams = {"Epochs":[100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000],
                       "LR":[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                       "MB":[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]}

    degrees = np.arange(1, poly_degrees + 1)
    lmbdas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    t0_values = np.arange(1, 11)
    t1_values = np.arange(10, 51, 5)


    #find_best_learning_schedule(t0_values, t1_values, N, noise, poly_degrees)

    #compare_variants_lr(N, noise, degrees[4::5], all_hyperparams[params], epoch=epochs)
    #compare_variants_epochs(N, noise, degrees[4::5], all_hyperparams[params][:9])
    #compare_variants_mb(N, noise, degrees[4::5], all_hyperparams[params], epoch=epochs)
    compare_variant_lr_lmbda(N, noise, degrees[4::5], all_hyperparams[params], lmbdas, epoch=epochs)


    '''
    poly_degrees = 15
    N = 1000
    noise = 0.1
    learning_rate = 0.001
    epochs = 500

    test_scores = np.zeros((5, poly_degrees))

    degrees = np.arange(1, poly_degrees + 1)
    variants = ["OLS", "standard fit", "decay fit", "RMSprop", "ADAM"]

    i = 0
    for variant in variants:
        j = 0
        for degree in degrees:
            
            x, y = generateData(N)
            X = computeDesignMatrix(x, y, degree)

            z = frankeFunction(x, y, noise=noise)

            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

            X_train = X_train[:,1:]
            X_test = X_test[:,1:]

            scaler = StandardScaler(with_mean=True, with_std=False)
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)


            X_train = np.c_[np.ones(X_train.shape[0]), X_train]
            X_test = np.c_[np.ones(X_test.shape[0]), X_test]

            if variant == "OLS":
                MODEL = OrdinaryLeastSquares(X_train, z_train)
                beta = MODEL.fit(*[X_train, z_train])
            else:
                MODEL = SGD(X_train, z_train, epochs=epochs)
            
                if variant == "standard fit":
                    beta = MODEL.fit(learning_rate)
            
                elif variant == "decay fit":
                    beta = MODEL.fit_with_decay(learning_rate)
                
                elif variant == "RMSprop":
                    beta = MODEL.RMSprop(learning_rate)

                elif variant == "ADAM":
                    beta = MODEL.ADAM(learning_rate)

            print(variant)

            z_predict = X_test @ beta
            error = MSE(z_test, z_predict)

            test_scores[i][j] = error

            j += 1
        
        i += 1
    
    plot_sgd_variants(test_scores, 
                      variants, 
                      N, 
                      noise, 
                      degrees, 
                      "comparison_sgd_variants_and_ols_LR=" + str(learning_rate) + "_Epochs=" + str(epochs))
    '''


        
            
                
