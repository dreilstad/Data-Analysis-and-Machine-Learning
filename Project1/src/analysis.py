import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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


class Analysis:
    '''The class contains static functions used in the analysis and assessment of regression methods.

    Contains a function for calculating the MSE score and the R2 score. In addition, various functions which
    plots the results from training and testing
    '''

    @staticmethod
    def R2(z, z_tilde):
        '''Function calculates the R2-score.

        Args:
            z (ndarray): a vector of size (Nx1) with the observed values
            z_tilde (ndarray): a vector of size (Nx1) with the predicted values
        Returns:
            float value bounded by 1 representing the quality of fit, 1 is a perfect fit
        '''

        return 1 - (np.sum((z - z_tilde) ** 2)) / (np.sum((z - np.mean(z)) ** 2))
    
    @staticmethod
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

    @staticmethod
    def plot_error_bias_variance_vs_complexity(MSE_scores, Bias, Variance, N_data, noise, degree, fig_name):
        '''Plots the bias-variance decomposition. Saves the plot as an image.

        Args:
            MSE_scores (ndarray): list of mse values
            Bias (ndarray): list of bias values
            Variance (ndarray): list of variance values
            N_data (int): size of dataset
            noise (float): amount of noise added
            degree (int): max polynomial degree
            fig_name (string): name of figure
        '''

        plt.plot(np.arange(1, len(MSE_scores) + 1), MSE_scores, label=r'$Error$')
        plt.plot(np.arange(1, len(Bias) + 1), Bias, label=r'$Bias^{2}$')
        plt.plot(np.arange(1, len(Variance) + 1), Variance, label=r'$Variance$')
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
        '''Plots MSE score as a function of complexity of the model. Saves the plot as an image.

        Args:
            MSE_training_scores (ndarray): list of mse values for the training set
            MSE_test_scores (ndarray): list of mse values for the test set
            N_data (int): size of dataset
            noise (float): amount of noise added
            degree (int): max polynomial degree
            fig_name (string): name of figure
        '''

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
    def plot_r2_vs_complexity(R2_training_scores, R2_test_scores, N_data, noise, degree, fig_name):
        '''Plots R2 score as a function of complexity of the model. Saves the plot as an image.

        Args:
            R2_training_scores (ndarray): list of r2 values for the training set
            R2_test_scores (ndarray): list of r2 values for the test set
            N_data (int): size of dataset
            noise (float): amount of noise added
            degree (int): max polynomial degree
            fig_name (string): name of figure
        '''

        plt.plot(np.arange(1, len(R2_training_scores) + 1), R2_training_scores, 'r-', label='R2 train')
        plt.plot(np.arange(1, len(R2_test_scores) + 1), R2_test_scores, 'b-', label='R2 test')
        plt.legend()
        plt.xlabel(r'Complexity of model (degree of polynomial)')
        plt.ylabel(r'R2 score')
        plt.grid()
        plt.tight_layout()
        save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=1-" + str(degree))
        plt.show()

    @staticmethod
    def plot_bootstrap_vs_kfold(MSE_test_scores, MSE_test_scores_boot, N_data, noise, degree, fig_name):
        '''Plots a comparison of MSE scores with bootstrap and k-fold cross validation as resmapling techniques. 
           Saves the plot as an image.

        Args:
            MSE_test_scores (ndarray): list of mse values for k-fold cross validation
            MSE_test_scores_boot (ndarray): list of mse values for bootstrap
            N_data (int): size of dataset
            noise (float): amount of noise added
            degree (int): max polynomial degree
            fig_name (string): name of figure
        '''

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
    def plot_confidence_intervals(betas, confidence_interval, N_data, noise, degree, fig_name):
        '''Plots the estimated 95% confidence intervals for the individual beta coefficients of a given model. 
           Saves the plot as an image.

        Args:
            betas (ndarray): a list of the beta coefficients
            confidence_interval (ndarray): a list of confidence interval corresponding to the beta coefficients
            N_data (int): size of dataset
            noise (float): amount of noise added
            degree (int): max polynomial degree
            fig_name (string): name of figure
        '''

        plt.scatter(np.arange(len(betas)), betas)
        plt.errorbar(np.arange(len(betas)), betas, yerr=confidence_interval, lw=1, fmt='none', capsize=3)

        plt.legend()
        plt.xlabel(r'$\beta_j$')
        plt.ylabel(r'$\beta_j value$')
        plt.grid()
        plt.tight_layout()
        save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=" + str(degree))
        plt.show()

    @staticmethod
    def plot_lambda_vs_complexity(R2_scores, degrees, lambdas, N_data, noise, fig_name, method):
        '''Plots a heatmap of the R2-score for all combination for a range of lambdas and polynomial degrees. 
           Saves the plot as image.

        Args:
            R2_scores (ndarray): a 2D array of r2 values for each lambda and polynomial degree
            degrees (ndarray): list of polynomial degrees
            lambdas (ndarray): list of lambdas
            N_data (int): size of dataset
            noise (float): amount of noise added
            fig_name (string): name of figure
            method (string): name of regression method used
        '''

        heatmap = sns.heatmap(R2_scores, 
                              annot=True,
                              cmap='Spectral',
                              xticklabels=np.around(np.log10(lambdas), 1), 
                              yticklabels=degrees, 
                              linewidths=0.5, 
                              annot_kws={"fontsize":8}, 
                              vmin=0.5, 
                              vmax=1)


        plt.xlabel(r'$log_{10}$($\lambda$)')
        plt.ylabel(r'Complexity of model (degree of polynomial)')

        if method == 'Ridge':
            plt.title(r'Ridge regression - R2 score')
        else:
            plt.title(r'Lasso regression - R2 score')

        plt.tight_layout()
        save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=1" + str(degrees[-1]))
        plt.show()

    @staticmethod
    def plot_ols_ridge_lasso(MSE_train_scores, MSE_test_scores, N_data, noise, degrees, fig_name):
        '''Plots a comparison of all regression method with corresponding MSE test and MSE trains scores.
           Saves the plot as image.

        Args:
            MSE_training_scores (ndarray): a 2D array of mse values for the training set in order OLS, Ridge and Lasso
            MSE_test_scores (ndarray): a 2D array of mse values for the test set in order OLS, Ridge and Lasso
            MSE_test_scores (ndarray): list of mse values for the test set
            N_data (int): size of dataset
            noise (float): amount of noise added
            degrees (int): list of polynomial degrees
            fig_name (string): name of figure
        '''

        plt.plot(degrees, MSE_train_scores[0], linestyle='--', color='red', label='OLS - Train')
        plt.plot(degrees, MSE_test_scores[0], linestyle='-', color='red', label='OLS - Test')

        plt.plot(degrees, MSE_train_scores[1], linestyle='--', color='blue', label='Ridge - Train')
        plt.plot(degrees, MSE_test_scores[1], linestyle='-', color='blue', label='Ridge - Test')

        plt.plot(degrees, MSE_train_scores[2], linestyle='--', color='orange', label='Lasso - Train')
        plt.plot(degrees, MSE_test_scores[2], linestyle='-', color='orange', label='Lasso - Test')
        
        plt.yscale('log')
        plt.legend()
        plt.xlabel(r'Complexity of model (degree of polynomial)')
        plt.ylabel(r'MSE score')
        plt.grid()
        plt.tight_layout()
        save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=1-" + str(len(degrees)))
        plt.show()