import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-dark')

PROJECT_ROOT_DIR = "Results" 
FIGURE_ID = "Results/FigureFiles" 

if not os.path.exists(PROJECT_ROOT_DIR): 
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID): 
    os.makedirs(FIGURE_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')


def R2(z, z_tilde):
    '''Function calculates the R2-score.

    Args:
        z (ndarray): a vector of size (Nx1) with the observed values
        z_tilde (ndarray): a vector of size (Nx1) with the predicted values
    Returns:
        float value bounded by 1 representing the quality of fit, 1 is a perfect fit
    '''

    return 1 - (np.sum((z - z_tilde) ** 2)) / (np.sum((z - np.mean(z)) ** 2))

def MSE(z, z_tilde):
    '''Function calculates the Mean Squared Error (MSE) score.

    Args:
        z (ndarray): a vector of size (Nx1) with the observed values
        z_tilde (ndarray): a vector of size (Nx1) with the predicted values
    Returns:
        float value of the error between the observed value and predicted values, 0 is no error
    '''

    N = np.size(z_tilde)
    return np.sum((z - z_tilde)**2) / N


def plot_compare_variants_epoch(test_scores, epochs, degree, N_data, noise, fig_name):
    """Plots a comparison MSE for all SGD variants as a function of the number fo epochs.
    Saves the plot as image.

    Args:
        test_scores (ndarray): a 2D array of mse values for the test set in order 
                               OLS, standard fit, decay fit, RMSprop and ADAM.
        epochs (ndarray): 1D array of epochs at which MSE is measured
        degrees (int): degree of polynomial
        N_data (int): size of dataset
        noise (float): amount of noise added
        fig_name (string): name of figure
    """

    variants = ["Standard fit", "Decay fit", "RMSprop", "ADAM", "OLS"]

    for i, test_score in enumerate(test_scores):
        plt.plot(epochs, test_score, label=variants[i])
    
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r'# of Epochs')
    plt.ylabel(r'MSE score')
    plt.grid()
    plt.tight_layout()
    save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=" + str(degree))
    plt.show()

def plot_compare_variants_lr(test_scores, learning_rates, degree, N_data, noise, fig_name):
    """Plots a comparison of MSE for all SGD variants as a function of the learning rate
    Saves the plot as image.

    Args:
        test_scores (ndarray): a 2D array of mse values for the test set in order 
                               OLS, standard fit, decay fit, RMSprop and ADAM.
        learning_rates (ndarray): 1D array of learning rates
        degrees (int): degree of polynomial
        N_data (int): size of dataset
        noise (float): amount of noise added
        fig_name (string): name of figure
    """

    variants = ["Standard fit", "Decay fit", "RMSprop", "ADAM", "OLS"]

    for i, test_score in enumerate(test_scores):
        plt.plot(np.around(np.log10(learning_rates), 1), test_score, label=variants[i])
    
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r'$log_{10}$($\eta$)')
    plt.ylabel(r'MSE score')
    plt.grid()
    plt.tight_layout()
    save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=" + str(degree))
    plt.show()

def plot_compare_variants_mb(test_scores, mini_bacthes, degree, N_data, noise, fig_name):
    """Plots a comparison of MSE for all SGD variants as a function of the size of the minibatches.
    Saves the plot as image.

    Args:
        test_scores (ndarray): a 2D array of mse values for the test set in order 
                               OLS, standard fit, decay fit, RMSprop and ADAM.
        mini_batches (ndarray): list of sizes of minibatches 
        degrees (int): list of polynomial degrees
        N_data (int): size of dataset
        noise (float): amount of noise added
        fig_name (string): name of figure
    """

    variants = ["Standard fit", "Decay fit", "RMSprop", "ADAM", "OLS"]

    for i, test_score in enumerate(test_scores):
        plt.plot(mini_bacthes, test_score, label=variants[i])
    
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r'size of mini-batches')
    plt.ylabel(r'MSE score')
    plt.grid()
    plt.tight_layout()
    save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=" + str(degree))
    plt.show()


def plot_learning_decay(scores, t1_values, degree, N, noise, fig_name):
    """Plots the MSE score for SGD with decay as a function of the t1 value.
    Saves the plot as an image.

    Args:
        scores (ndarray): 1D array of MSE test scores
        t1_values (ndarray): 1D array of t1 values 
        degree (int): degree of polynomial
        N (int): size of dataset
        noise (float): amount of noise added
        fig_name (str): name of figure
    """

    
    plt.plot([str(t1) for t1 in t1_values], scores, label="Decay fit")
    
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r'$t_{1}$-value')
    plt.ylabel(r'MSE score')
    plt.grid()
    plt.tight_layout()
    save_fig(fig_name + "_N=" + str(N) + "_Noise=" + str(noise) + "_Degree=" + str(degree))
    plt.show()


def plot_lmbda_vs_lr_heatmap(scores, learning_rates, lmbdas, degree, variant, N, noise, measurement, fig_name):
    """Plots the scores for SGD variants as a function of the learning rate and regularization value.
    Saves the plot as an image.

    Args:
        scores (ndarray): 2D array of MSE test scores
        learning_rates (ndarray): 1D array of learning rates
        lmbdas (ndarray): 1D array of regularization values
        degree (int): degree of polynomial
        variant (str): stochastic gradient descent variant
        N (int): size of dataset
        noise (float): amount of noise added
        measurement (str): metric type
        fig_name (str): name of figure
    """

    heatmap = sns.heatmap(scores, 
                            annot=True,
                            cmap='Spectral',
                            xticklabels=np.around(np.log10(lmbdas), 1), 
                            yticklabels=np.around(np.log10(learning_rates), 1), 
                            linewidths=0.5, 
                            annot_kws={"fontsize":12})


    plt.xlabel(r'$log_{10}$($\lambda$)')
    plt.ylabel(r'$log_{10}$($\eta$)')
    plt.title(variant + " - " + measurement + " score")
    plt.tight_layout()
    save_fig(fig_name + "_Variant=" + variant + "_N=" + str(N) + "_Noise=" + str(noise) + "_Degree=" + str(degree) + "_" + measurement)
    plt.show()

def plot_scores_vs_neurons(scores, neurons_layer, epochs, size_batch, learning_rate, lmbda, optimizer):
    """Plots the scores for feed forward neural network as a function of number of neurons in the hidden layer.
    Saves the plot as an image.

    Args:
        scores (ndarray): 2D array of test scores for each activation function
        neurans_layer (ndarray): 1D array of number of neurons
        epochs (int): number of epochs
        size_batch (int): size of minibatches
        learning_rate (float): learning rate
        lmbda (float): regularization value
        optimizer (str): stochastic gradient descent optimizer
    """   
    
    plt.plot(neurons_layer, scores[0], label="Sigmoid")
    plt.plot(neurons_layer, scores[1], label="Tanh")
    plt.plot(neurons_layer, scores[2], label="RELU")
    plt.plot(neurons_layer, scores[3], label="LeakyRELU")

    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.legend()
    plt.title("FFNN - " + optimizer + " - Accuracy test score")
    plt.ylabel(r"Accuracy score")
    plt.xlabel(r"# of neurons in layer")
    plt.grid()
    plt.tight_layout()
    save_fig("ffnn_accuracy_vs_neurons_Epochs={}_sizeMB={}_variant={}_lr={}_lmbda={}.png".format(epochs, size_batch, optimizer, learning_rate, lmbda))
    plt.show()

def plot_scores_vs_layers(scores, nr_layers, neurons, epochs, size_batch, learning_rate, lmbda, optimizer):
    """Plots the scores for feed forward neural network as a function of number of hidden layers.
    Saves the plot as an image.

    Args:
        scores (ndarray): 2D array of test scores for each activation function
        neurans_layer (ndarray): 1D array of number of neurons
        epochs (int): number of epochs
        size_batch (int): size of minibatches
        learning_rate (float): learning rate
        lmbda (float): regularization value
        optimizer (str): stochastic gradient descent optimizer
    """   
    
    plt.plot(nr_layers, scores[0], label="Sigmoid")
    plt.plot(nr_layers, scores[1], label="Tanh")
    plt.plot(nr_layers, scores[2], label="RELU")
    plt.plot(nr_layers, scores[3], label="LeakyRELU")

    plt.xticks(nr_layers)
    plt.legend()
    plt.title("FFNN - " + optimizer + " - Accuracy test score")
    plt.ylabel(r"Accuracy score")
    plt.xlabel(r"# of hidden layers")
    plt.grid()
    plt.tight_layout()
    save_fig("ffnn_accuracy_vs_layers_Neurons={}_Epochs={}_sizeMB={}_variant={}_lr={}_lmbda={}.png".format(neurons, epochs, size_batch, optimizer, learning_rate, lmbda))
    plt.show()

def plot_ffnn_accuracy_lr_vs_lmbda(scores, xlabels, ylabels, optimizer, fig_name):
    """Plots the accuracy scores for feed forward neural network as a function of the learning rate and regularization value.
    Saves the plot as an image.

    Args:
        scores (ndarray): 2D array of test scores
        xlabels (ndarray): 1D array of learning rates
        ylabels (ndarray): 1D array of regularization values
        degree (int): degree of polynomial
        optimizer (str): stochastic gradient descent variant
        fig_name (str): name of figure
    """

    sns.heatmap(scores, cmap='Spectral', xticklabels = xlabels, yticklabels = ylabels, annot=True, vmin=0.2, vmax=1.0)
    plt.title("FFNN - " + optimizer + " - Accuracy test score")
    plt.ylabel(r"$log_{10}(\eta)$")
    plt.xlabel(r"$log_{10}(\lambda)$")
    save_fig(fig_name + "_variant={}".format(optimizer))
    plt.show()

def plot_logreg_accuracy_lr_vs_lmbda(scores, xlabels, ylabels, method, fig_name):
    """Plots the accuracy scores for logistic regression as a function of the learning rate and regularization value.
    Saves the plot as an image.

    Args:
        scores (ndarray): 2D array of test scores
        xlabels (ndarray): 1D array of learning rates
        ylabels (ndarray): 1D array of regularization values
        method (int): own logistic regression or scikit's sgdclassifier
        fig_name (str): name of figure
    """

    sns.heatmap(scores, cmap='Spectral', xticklabels = xlabels, yticklabels = ylabels, annot=True, vmin=0.2, vmax=1.0)
    plt.title(method + " - standard SGD - Accuracy test score")
    plt.ylabel(r"$log_{10}(\eta)$")
    plt.xlabel(r"$log_{10}(\lambda)$")
    save_fig(fig_name + "_variant=standard")
    plt.show()
