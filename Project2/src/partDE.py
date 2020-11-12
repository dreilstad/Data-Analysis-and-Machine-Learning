import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tools as tools
import logistic as log
import analysis as an

from sklearn import datasets
from neural_network import FFNN, Layer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

import seaborn as sns
plt.style.use('seaborn-dark')


def logreg_lr_vs_lmbda(X_train, X_test, y_test, y_train_onehot, epochs, size_batch):
    """
    Uses multinomial logistic regression to fit parameters for different classes to the dataset for different 
    combinations of learning rates and regularization values. Calculate sthe accuracy of the model and plots the
    result as a heatmap.
    """

    # add intercept
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    learning_rates = np.logspace(-5, 0, 6)
    lmbdas = np.logspace(-7, 0, 8)

    scores = np.zeros((len(learning_rates), len(lmbdas)))
    scikit_scores = np.zeros((len(learning_rates), len(lmbdas)))

    for i, learning_rate in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbdas):
            
            
            model = log.LogisticRegression(X_train, y_train, y_train_onehot, n_categories, learning_rate, lmbda, epochs, size_batch)
            beta = model.fit()
            prediction, probabilites = model.predict(X_test, beta)
            accuracy = np.round(model.accuracy(y_test, prediction), 2)


            scikit_model = SGDClassifier(loss="log", penalty="l2", alpha=lmbda, fit_intercept=False, 
                                         max_iter=epochs, learning_rate="constant", eta0=learning_rate)
            
            indices = np.arange(X_train.shape[0])
            shuffler = np.random.default_rng()
            for epoch in range(epochs):
                shuffler.shuffle(indices)
                for batch in range(1, (X_train.shape[0] // size_batch) + 1):
                    batch_indices = np.random.choice(indices, size_batch, replace=True)
                    scikit_model.partial_fit(X_train[batch_indices], y_train[batch_indices], classes=np.unique(labels))
            
            scikit_prediction = scikit_model.predict(X_test)
            scikit_accuracy = float(sum(scikit_prediction == y_test) / len(y_test))

            scores[i][j] = accuracy
            scikit_scores[i][j] = scikit_accuracy

            print("Learning rate  = {}, Lambda = {}, Accuracy = {} Scikit-Accuracy = {}" .format(learning_rate, lmbda, accuracy, scikit_accuracy))

    
    xlabels = [np.log10(x) for x in lmbdas]
    ylabels = [np.log10(y) for y in learning_rates] 

    an.plot_logreg_accuracy_lr_vs_lmbda(scores, xlabels, ylabels, "Logistic regression", "logreg_accuracy_lr_vs_lmbda_epochs={}".format(epochs))
    an.plot_logreg_accuracy_lr_vs_lmbda(scikit_scores, xlabels, ylabels, "Scikit SGDClassifier", "sgdclassifier_accuracy_lr_vs_lmbda_epochs={}".format(epochs))


def ffnn_accuracy_vs_neurons(epochs, size_batch, optimizer, n_categories):
    """
    Trains the neural network for different amounts of neurons in a single hidden layer, and calculates the accuracy. 
    Plots the result.
    """

    neurons_layer = [10, 20, 30, 40, 50, 100, 150, 200]
    learning_rate = 0.000001
    lmbda = 0.00001

    scores = np.zeros((4, len(neurons_layer)))
    
    for i, neurons in enumerate(neurons_layer):

        ffnn_sig, ffnn_tanh, ffnn_relu, ffnn_leakyrelu = init_neural_networks(optimizer, 1, neurons)

        ffnn_sig.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
        prediction_sig = ffnn_sig.predict(X_test).flatten()
        accuracy_sig = ffnn_sig.accuracy(y_test, prediction_sig)


        ffnn_tanh.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
        prediction_tanh = ffnn_tanh.predict(X_test).flatten()
        accuracy_tanh = ffnn_tanh.accuracy(y_test, prediction_tanh)


        ffnn_relu.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
        prediction_relu = ffnn_relu.predict(X_test).flatten()
        accuracy_relu = ffnn_relu.accuracy(y_test, prediction_relu)


        ffnn_leakyrelu.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
        prediction_leakyrelu = ffnn_leakyrelu.predict(X_test).flatten()
        accuracy_leakyrelu = ffnn_leakyrelu.accuracy(y_test, prediction_leakyrelu)

        scores[0][i] = accuracy_sig
        scores[1][i] = accuracy_tanh
        scores[2][i] = accuracy_relu
        scores[3][i] = accuracy_leakyrelu


        print("Accuracy vs # of neurons: Neurons = {}" .format(neurons))
        print("                          Accuracy sigmoid = {}".format(accuracy_sig))
        print("                          Accuracy tanh = {}".format(accuracy_tanh))
        print("                          Accuracy relu = {}".format(accuracy_relu))
        print("                          Accuracy leakyrelu = {}".format(accuracy_leakyrelu))

       
    an.plot_scores_vs_neurons(scores, neurons_layer, epochs, size_batch, learning_rate, lmbda, optimizer)


def init_neural_networks(optimizer, layers, neurons):
    """
    Creates a dense neural network with specified number of hidden layers, all constisting of specified number of neurons, 
    with the different activation functions.
    """

    ffnn_sig = FFNN(optimizer=optimizer)
    ffnn_sig.add_layer(Layer(X_train.shape[1], neurons, "sigmoid"))
    for j in range(1, layers):
            ffnn_sig.add_layer(Layer(neurons, neurons, "sigmoid"))
    ffnn_sig.add_layer(Layer(neurons, n_categories, "softmax"))


    ffnn_tanh = FFNN(optimizer=optimizer)
    ffnn_tanh.add_layer(Layer(X_train.shape[1], neurons, "tanh"))
    for j in range(1, layers):
            ffnn_tanh.add_layer(Layer(neurons, neurons, "tanh"))
    ffnn_tanh.add_layer(Layer(neurons, n_categories, "softmax"))


    ffnn_relu = FFNN(optimizer=optimizer)
    ffnn_relu.add_layer(Layer(X_train.shape[1], neurons, "relu"))
    for j in range(1, layers):
            ffnn_relu.add_layer(Layer(neurons, neurons, "relu"))
    ffnn_relu.add_layer(Layer(neurons, n_categories, "softmax"))


    ffnn_leakyrelu = FFNN(optimizer=optimizer)
    ffnn_leakyrelu.add_layer(Layer(X_train.shape[1], neurons, "leakyrelu"))
    for j in range(1, layers):
            ffnn_leakyrelu.add_layer(Layer(neurons, neurons, "leakyrelu"))
    ffnn_leakyrelu.add_layer(Layer(neurons, n_categories, "softmax"))


    return ffnn_sig, ffnn_tanh, ffnn_relu, ffnn_leakyrelu

def ffnn_accuracy_vs_layers(epochs, size_batch, optimizer, n_categories):
    """
    Trains the neural network for different amounts of hidden layers, and calculates the accuracy. 
    Plots the result.
    """

    nr_layers = [1, 2, 3, 4, 5]
    learning_rate = 0.00001
    lmbda = 0.00001
    neurons = 50

    scores = np.zeros((4, len(nr_layers)))


    for i, layers in enumerate(nr_layers):

        ffnn_sig, ffnn_tanh, ffnn_relu, ffnn_leakyrelu = init_neural_networks(optimizer, layers, neurons)

        ffnn_sig.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
        prediction_sig = ffnn_sig.predict(X_test).flatten()
        accuracy_sig = ffnn_sig.accuracy(y_test, prediction_sig)

        ffnn_tanh.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
        prediction_tanh = ffnn_tanh.predict(X_test).flatten()
        accuracy_tanh = ffnn_tanh.accuracy(y_test, prediction_tanh)

        ffnn_relu.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
        prediction_relu = ffnn_relu.predict(X_test).flatten()
        accuracy_relu = ffnn_relu.accuracy(y_test, prediction_relu)
        
        ffnn_leakyrelu.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
        prediction_leakyrelu = ffnn_leakyrelu.predict(X_test).flatten()
        accuracy_leakyrelu = ffnn_leakyrelu.accuracy(y_test, prediction_leakyrelu)

        scores[0][i] = accuracy_sig
        scores[1][i] = accuracy_tanh
        scores[2][i] = accuracy_relu
        scores[3][i] = accuracy_leakyrelu


        print("Accuracy vs # of layers:  Layers = {}" .format(layers))
        print("                          Accuracy sigmoid = {}".format(accuracy_sig))
        print("                          Accuracy tanh = {}".format(accuracy_tanh))
        print("                          Accuracy relu = {}".format(accuracy_relu))
        print("                          Accuracy leakyrelu = {}".format(accuracy_leakyrelu))

       
    an.plot_scores_vs_layers(scores, nr_layers, neurons, epochs, size_batch, learning_rate, lmbda, optimizer)


def ffnn_lr_vs_lmbda(neurons, epochs, size_batch, optimizer, n_categories):
    """
    Trains the neural network for each combination of learning rates and regularization values, and calculates the accuracy.
    Plots the result as a heatmap.
    """

    learning_rates = np.logspace(-5, -2, 4)
    lmbdas = np.logspace(-6, -3, 4)

    layers = 1
    neurons = 64

    scores = np.zeros((4, len(learning_rates), len(lmbdas)))
    scikit_scores = np.zeros((len(learning_rates), len(lmbdas)))

    for i, learning_rate in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbdas):

            ffnn_sig, ffnn_tanh, ffnn_relu, ffnn_leakyrelu = init_neural_networks(optimizer, layers, neurons)

            ffnn_sig.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
            prediction_sig = ffnn_sig.predict(X_test).flatten()
            accuracy_sig = ffnn_sig.accuracy(y_test, prediction_sig)

            ffnn_tanh.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
            prediction_tanh = ffnn_tanh.predict(X_test).flatten()
            accuracy_tanh = ffnn_tanh.accuracy(y_test, prediction_tanh)

            ffnn_relu.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
            prediction_relu = ffnn_relu.predict(X_test).flatten()
            accuracy_relu = ffnn_relu.accuracy(y_test, prediction_relu)
            
            ffnn_leakyrelu.train(X_train, y_train_onehot, learning_rate, epochs, size_batch=size_batch, lmbda=lmbda)
            prediction_leakyrelu = ffnn_leakyrelu.predict(X_test).flatten()
            accuracy_leakyrelu = ffnn_leakyrelu.accuracy(y_test, prediction_leakyrelu)

            scores[0][i][j] = accuracy_sig
            scores[1][i][j] = accuracy_tanh
            scores[2][i][j] = accuracy_relu
            scores[3][i][j] = accuracy_leakyrelu



            print("\nLearning rate vs lambda:  Learning rate = {}, Lambda = {}" .format(learning_rate, lmbda))
            print("                          Accuracy sigmoid = {}".format(accuracy_sig))
            print("                          Accuracy tanh = {}".format(accuracy_tanh))
            print("                          Accuracy relu = {}".format(accuracy_relu))
            print("                          Accuracy leakyrelu = {}".format(accuracy_leakyrelu))

    xlabels = [np.log10(x) for x in lmbdas]
    ylabels = [np.log10(y) for y in learning_rates] 

    an.plot_ffnn_accuracy_lr_vs_lmbda(scores[0], xlabels, ylabels, optimizer, "ffnn_accuracy_lr_vs_lmbda_act=sigmoid_Epochs={}_sizeMB={}".format(epochs, size_batch))
    an.plot_ffnn_accuracy_lr_vs_lmbda(scores[1], xlabels, ylabels, optimizer, "ffnn_accuracy_lr_vs_lmbda_act=tanh_Epochs={}_sizeMB={}".format(epochs, size_batch))
    an.plot_ffnn_accuracy_lr_vs_lmbda(scores[2], xlabels, ylabels, optimizer, "ffnn_accuracy_lr_vs_lmbda_act=relu_Epochs={}_sizeMB={}".format(epochs, size_batch))
    an.plot_ffnn_accuracy_lr_vs_lmbda(scores[3], xlabels, ylabels, optimizer, "ffnn_accuracy_lr_vs_lmbda_act=leakyrelu_Epochs={}_sizeMB={}".format(epochs, size_batch))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Project 2 - Part D & E - How to use script', add_help=False)
    parser._action_groups.pop()
    possible_args = parser.add_argument_group('possible arguments')


    possible_args.add_argument('-o', '--optimizer', 
                               type=str, 
                               required=False,
                               choices=['standard', 'RMSprop', 'ADAM'],
                               help='Choose optimizer')

    possible_args.add_argument('-e', '--epochs', 
                               type=int, 
                               required=True,
                               default=100,
                               help='Specify number of epochs')

    possible_args.add_argument('-n', '--neurons', 
                               type=int, 
                               required=False,
                               default=64,
                               help='Specify number of neurons')

    possible_args.add_argument('-mb', '--size_minibatch', 
                               type=int, 
                               required=True,
                               default=10,
                               help='Specify size of the minibatches')

    possible_args.add_argument('-f', '--function', 
                               type=str, 
                               required=True,
                               choices=['logistic', 'neurons', 'layers', 'heatmap'],
                               help='Choose function')    
    
    possible_args.add_argument('-h', '--help',
                               action='help',
                               help='Helpful message showing flags and usage of code for part A')

    args = parser.parse_args()

    optimizer = args.optimizer
    epochs = args.epochs
    size_batch = args.size_minibatch
    neurons = args.neurons
    function = args.function
    
    dataset = datasets.load_digits()
    images = dataset.images
    labels = dataset.target

    n_categories = len(np.unique(labels))
    n_images, image_width, image_height = images.shape
    image_size = image_width * image_height

    images = images.reshape(n_images, image_size)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_train_onehot = tools.to_categorical_numpy(y_train)
    y_test_onehot = tools.to_categorical_numpy(y_test)

    if function == "logistic":
        logreg_lr_vs_lmbda(X_train, X_test, y_test, y_train_onehot, epochs, size_batch)
    elif function == "neurons":
        ffnn_accuracy_vs_neurons(epochs, size_batch, optimizer, n_categories)
    elif function == "layers":
        ffnn_accuracy_vs_layers(epochs, size_batch, optimizer, n_categories)
    elif function == "heatmap":
        ffnn_lr_vs_lmbda(neurons, epochs, size_batch, optimizer, n_categories)