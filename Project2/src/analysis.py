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



def plot_sgd_variants(test_scores, labels, N_data, noise, degrees, fig_name):
    """Plots a comparison of all SGD variants.
    Saves the plot as image.

    Args:
        test_scores (ndarray): a 2D array of mse values for the test set in order 
                               OLS, standard fit, decay fit, RMSprop and ADAM.
        labels (ndarray): list of mse values for the test set
        N_data (int): size of dataset
        noise (float): amount of noise added
        degrees (int): list of polynomial degrees
        fig_name (string): name of figure
    """

    for i, test_score in enumerate(test_scores):
        plt.plot(degrees, test_score, label=labels[i])
    
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r'Complexity of model (degree of polynomial)')
    plt.ylabel(r'MSE score')
    plt.grid()
    plt.tight_layout()
    save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=1-" + str(len(degrees)))
    plt.show()

def plot_compare_variants_epoch(test_scores, epochs, degree, N_data, noise, fig_name):
    """Plots a comparison of all SGD variants.
    Saves the plot as image.

    Args:
        test_scores (ndarray): a 2D array of mse values for the test set in order 
                               OLS, standard fit, decay fit, RMSprop and ADAM.
        labels (ndarray): list of mse values for the test set
        N_data (int): size of dataset
        noise (float): amount of noise added
        degrees (int): list of polynomial degrees
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
    """Plots a comparison of all SGD variants.
    Saves the plot as image.

    Args:
        test_scores (ndarray): a 2D array of mse values for the test set in order 
                               OLS, standard fit, decay fit, RMSprop and ADAM.
        labels (ndarray): list of mse values for the test set
        N_data (int): size of dataset
        noise (float): amount of noise added
        degrees (int): list of polynomial degrees
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
    """Plots a comparison of all SGD variants.
    Saves the plot as image.

    Args:
        test_scores (ndarray): a 2D array of mse values for the test set in order 
                               OLS, standard fit, decay fit, RMSprop and ADAM.
        labels (ndarray): list of mse values for the test set
        N_data (int): size of dataset
        noise (float): amount of noise added
        degrees (int): list of polynomial degrees
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

def plot_heatmap_epochs(score, epochs, degrees, variant, N_data, noise, fig_name):
    heatmap = sns.heatmap(score, 
                          annot=True,
                          cmap='Spectral',
                          xticklabels=epochs, 
                          yticklabels=degrees, 
                          linewidths=0.5, 
                          annot_kws={"fontsize":8}, 
                          vmin=0.5, 
                          vmax=1)


    plt.xlabel(r'# of Epochs')
    plt.ylabel(r'Complexity of model (degree of polynomial)')
    plt.title(variant + " - R2 score")
    plt.tight_layout()
    save_fig(fig_name + "_Variant=" + variant + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=1" + str(degrees[-1]))
    plt.show()


def plot_heatmap(scores, labels, variant, N, noise, vmin, vmax, fig_name):

    y_label, x_label = labels.keys()
    
    heatmap = sns.heatmap(scores, 
                          annot=True,
                          cmap='Spectral',
                          xticklabels=labels[x_label], 
                          yticklabels=labels[y_label], 
                          linewidths=0.5, 
                          annot_kws={"fontsize":8},
                          vmin=vmin, 
                          vmax=vmax)


    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(variant)
    plt.tight_layout()
    save_fig(fig_name + "_Variant=" + variant + "_N=" + str(N) + "_Noise=" + str(noise))
    plt.show()

def plot_lmbda_vs_lr_heatmap(scores, learning_rates, lmbdas, degree, variant, N, noise, fig_name):
    heatmap = sns.heatmap(scores, 
                          annot=True,
                          cmap='Spectral',
                          xticklabels=np.around(np.log10(lmbdas), 1), 
                          yticklabels=np.around(np.log10(learning_rates), 1), 
                          linewidths=0.5, 
                          annot_kws={"fontsize":8}, 
                          vmin=0.5, 
                          vmax=1)


    plt.xlabel(r'$log_{10}$($\lambda$)')
    plt.ylabel(r'$log_{10}$($\eta$)')
    plt.title(variant + " - R2 score")
    plt.tight_layout()
    save_fig(fig_name + "_Variant=" + variant + "_N=" + str(N) + "_Noise=" + str(noise) + "_Degree=" + str(degree))
    plt.show()