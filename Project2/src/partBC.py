import sys
import argparse
import numpy as np
import analysis as an
import tools as tools

from sgd import SGD
from ols import OrdinaryLeastSquares
from ridge import Ridge
from neural_network import FFNN, Layer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def ffnn_lmbda_vs_lr(N, noise, degree, neurons, epochs, size_batch, activation, optimizer):
    """
    Trains the neural network for each combination of learning rates and regularization values, and calculates the accuracy.
    Plots the result as a heatmap.
    """        

    learning_rates = np.logspace(-6, 1, 7)
    lmbdas = np.logspace(-6, 1, 7)
        

    x, y = tools.generateData(N)
    X = tools.computeDesignMatrix(x, y, degree)
    X = X[:,1:]
    z = tools.frankeFunction(x, y, noise=noise)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    z_train = (z_train - np.min(z_train)) / (np.max(z_train) - np.min(z_train))
    z_test = (z_test - np.min(z_test)) / (np.max(z_test) - np.min(z_test))
    
    MSE = np.zeros((len(learning_rates), len(lmbdas)))
    R2 = np.zeros((len(learning_rates), len(lmbdas)))
    scikit_MSE = np.zeros((len(learning_rates), len(lmbdas)))
    scikit_R2 = np.zeros((len(learning_rates), len(lmbdas)))

    n_categories = 1
    threshold = 10

    for i, learning_rate in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbdas):

            ffnn = FFNN(optimizer=optimizer)
            ffnn.add_layer(Layer(X_train.shape[1], neurons, activation))
            ffnn.add_layer(Layer(neurons, n_categories, "none"))
            
            ffnn.train(X_train, z_train, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
            z_pred = ffnn.predict(X_test)

            MSE[i][j] = ffnn.MSE(z_test, z_pred.flatten())

            print("Gridsearch: Learning rate = {}, Lambda = {}".format(learning_rate, lmbda))
            print("            own MSE = {}".format(MSE[i][j]))



    MSE = np.where(MSE > threshold, np.nan, MSE)

    an.plot_lmbda_vs_lr_heatmap(MSE, learning_rates, lmbdas, degree, "FFNN - " + optimizer, 
                                N, noise, "MSE", "ffnn_heatmap_lmbda_vs_lr_Act=" + 
                                str(activation) + "_Epochs=" + str(epochs) + 
                                "_sizeMB=" + str(size_batch) + "_neurons=" + str(neurons))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Project 2 - Part B & C - How to use script', add_help=False)
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

    possible_args.add_argument('-a', '--activation', 
                               type=str, 
                               required=True,
                               choices=['sigmoid', 'tanh', 'relu', 'leakyrelu'],
                               help='Choose activation function')

    possible_args.add_argument('-o', '--optimizer', 
                               type=str, 
                               required=True,
                               choices=['standard', 'RMSprop', 'ADAM'],
                               help='Choose optimizer')

    possible_args.add_argument('-e', '--epochs', 
                               type=int, 
                               required=False,
                               default=100,
                               help='Specify number of epochs')

    possible_args.add_argument('-n', '--neurons', 
                               type=int, 
                               required=False,
                               default=50,
                               help='Specify learning rate')

    possible_args.add_argument('-mb', '--size_minibatch', 
                               type=int, 
                               required=False,
                               default=10,
                               help='Specify size of the minibatches')
    
    possible_args.add_argument('-h', '--help',
                               action='help',
                               help='Helpful message showing flags and usage of code for part A')

    args = parser.parse_args()

    N = args.N_datapoints
    noise = args.noise
    poly_degrees = args.degree
    activation = args.activation
    optimizer = args.optimizer
    epochs = args.epochs
    size_batch = args.size_minibatch
    neurons = args.neurons

    ffnn_lmbda_vs_lr(N, noise, poly_degrees, neurons, epochs, size_batch, activation, optimizer)
