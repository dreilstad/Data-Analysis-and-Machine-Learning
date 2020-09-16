import os
import numpy as np
import matplotlib.pyplot as plt
from regression import Regression

PROJECT_ROOT_DIR = "Results" 
FIGURE_ID = "Results/FigureFiles" 
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR): 
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID): 
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID): 
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')


class Analysis:

    @staticmethod
    def R2(z, z_tilde):
        return 1 - (np.sum((z - z_tilde) ** 2)) / (np.sum((z - np.mean(z)) ** 2))
    
    @staticmethod
    def MSE(z, z_tilde):
        N = np.size(z_tilde)
        return np.sum((z - z_tilde)**2) / N

    @staticmethod
    def Bias(z, z_tilde):
        return np.mean((z - np.mean(z_tilde)) ** 2)

    @staticmethod
    def Variance(z_tilde):
        return np.mean(np.var(z_tilde))

    @staticmethod
    def beta_coeff_variance(X, z, z_predict):
        N, p = X.shape
        variance = (1/(N-p-1))*sum((z - z_predict)**2)
        return np.diagonal(np.linalg.pinv(X)) * variance

    @staticmethod
    def plot_error_bias_variance_vs_complexity(MSE_scores, Bias, Variance, N_data, noise, degree, fig_name):

        plt.plot(np.arange(1, len(MSE_scores) + 1), MSE_scores, label='Error')
        plt.plot(np.arange(1, len(Bias) + 1), Bias, label='Bias')
        plt.plot(np.arange(1, len(Variance) + 1), Variance, label='Variance')
        plt.yscale('log')
        plt.legend()
        plt.xlabel(r'Complexity of model (degree of polynomial)')
        plt.ylabel(r'MSE score')
        plt.grid()
        plt.tight_layout()
        save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=1-" + str(degree))
        plt.show()

    @staticmethod
    def plot_mse_vs_complexity(MSE_training_scores, MSE_test_scores, N_data, noise, degree, fig_name):

        plt.plot(np.arange(1, len(MSE_training_scores) + 1), MSE_training_scores, 'r-', label='MSE train')
        plt.plot(np.arange(1, len(MSE_test_scores) + 1), MSE_test_scores, 'b-', label='MSE test')
        plt.yscale('log')
        plt.legend()
        plt.xlabel(r'Complexity of model (degree of polynomial)')
        plt.ylabel(r'MSE score')
        plt.grid()
        plt.tight_layout()
        save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=1-" + str(degree))
        plt.show()

    @staticmethod
    def plot_bootstrap_vs_kfold(MSE_test_scores, MSE_test_scores_boot, N_data, noise, degree, fig_name):

        plt.plot(np.arange(1, len(MSE_test_scores) + 1), MSE_test_scores, 'r-', label='MSE test - KFold')
        plt.plot(np.arange(1, len(MSE_test_scores_boot) + 1), MSE_test_scores_boot, 'b-', label='MSE test - Bootstrap')
        plt.yscale('log')
        plt.legend()
        plt.xlabel(r'Complexity of model (degree of polynomial)')
        plt.ylabel(r'MSE score')
        plt.grid()
        plt.tight_layout()
        save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=1-" + str(degree))
        plt.show()


    @staticmethod
    def plot_confidence_intervals(Model, degree):
        """
        Function for finding the estimated confidence intervals of a given models beta-parameters,
        and makes a plot of the parameters with confidence intervals corresponing to
        a 95% confidence interval.
        """

        beta = Model.beta
        variance_beta = beta_coeff_variance(Model.X_test, Model.z, Model.z_predict)
        confidence_interval = 1.96 * np.sqrt(variance_beta)


        plt.errorbar(np.arange(len(beta)), beta, confidence_interval, fmt="b.", capsize=3, label=r'$\beta_j \pm 1.96 \sigma$')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend()
        plt.xlabel(r'index $j$')
        plt.ylabel(r'$\beta_j$')
        plt.grid()
        save_fig("confidence_interval_beta_Degree=1-" + str(degree))
        plt.show()