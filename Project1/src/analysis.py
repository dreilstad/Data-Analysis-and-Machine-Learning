import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from regression import Regression
plt.style.use('seaborn-dark')

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
    def plot_error_bias_variance_vs_complexity(MSE_scores, Bias, Variance, N_data, noise, degree, fig_name):

        plt.plot(np.arange(1, len(MSE_scores) + 1), MSE_scores, label=r'$Error$')
        plt.plot(np.arange(1, len(Bias) + 1), Bias, label=r'$Bias^{2}$')
        plt.plot(np.arange(1, len(Variance) + 1), Variance, label=r'$Variance$')
        #plt.plot(np.arange(1, len(Variance) + 1), Bias + Variance, linestyle='--', label=r'$Bias^{2} + Variance$')
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
    def plot_r2_vs_complexity(R2_training_scores, R2_test_scores, N_data, noise, degree, fig_name):

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
    def plot_mse_vs_lambda(Lambdas, MSE_training_scores, MSE_test_scores, N_data, noise, degree, fig_name):

        plt.plot(np.log10(Lambdas), MSE_training_scores, 'r-', label='MSE train - Ridge')
        plt.plot(np.log10(Lambdas), MSE_test_scores, 'b-', label='MSE test - Ridge')
        plt.yscale('log')
        plt.legend()
        plt.xlabel(r'$log_{10}$($\lambda$)')
        plt.ylabel(r'MSE score')
        plt.grid()
        plt.tight_layout()
        save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise))
        plt.show()

    @staticmethod
    def plot_beta_ci_vs_lambda(Betas, confidence_intervals, lambdas):
        

        f, ax = plt.subplots()

        for i in range(Betas.shape[1]):

            print(Betas[i,:])
            print(confidence_intervals[i,:])

            plt.errorbar(np.log10(lambdas), Betas[i,:], yerr=confidence_intervals[i,:], fmt='-o', capsize=3, label=r'$\beta_{} \pm 1.96 \sigma$'.format(i))

        plt.legend()
        plt.xlabel(r'$log_{10}$($\lambda$)')
        plt.ylabel(r'$\beta_j$')
        plt.grid()
        plt.show()


    @staticmethod
    def plot_confidence_intervals(betas, confidence_interval, N_data, noise, degree, fig_name):
        """
        Function for finding the estimated confidence intervals of a given models beta-parameters,
        and makes a plot of the parameters with confidence intervals corresponing to
        a 95% confidence interval.
        """

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
    def plot_ridge_bias_variance(MSE_scores, Bias, Variance, Lambdas, N_data, noise, degree, fig_name):

        plt.plot(np.log10(Lambdas), MSE_scores, label='Ridge - Error')
        plt.plot(np.log10(Lambdas), Bias, label='Ridge - Bias')
        plt.plot(np.log10(Lambdas), Variance, label='Ridge - Variance')
        plt.yscale('log')
        plt.legend()
        plt.xlabel(r'$log_{10}$($\lambda$)')
        plt.ylabel(r'MSE score')
        plt.grid()
        plt.tight_layout()
        #save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=" + str(degree))
        plt.show()

    @staticmethod
    def plot_lambda_vs_complexity(R2_scores, degrees, lambdas, N_data, noise, fig_name, method):

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


    @staticmethod
    def plot_ols_ridge_lasso_r2(R2_OLS, R2_Ridge, R2_Lasso, N_data, noise, degrees, fig_name):

        plt.plot(degrees, R2_OLS, linestyle='-', color='red', label='R2 - OLS')
        plt.plot(degrees, R2_Ridge, linestyle='-', color='blue', label='R2 - Ridge')
        plt.plot(degrees, R2_Lasso, linestyle='-', color='orange', label='R2 - Lasso')
        plt.yscale('log')
        plt.legend()
        plt.xlabel(r'Complexity of model (degree of polynomial)')
        plt.ylabel(r'MSE score')
        plt.grid()
        plt.tight_layout()
        save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise) + "_Degree=1-" + str(len(degrees)))
        plt.show()

    @staticmethod
    def plot_ols_ridge_lasso_lambda(MSE_train_scores, MSE_test_scores, N_data, noise, lambdas, fig_name):
        plt.plot(np.log10(lambdas), MSE_train_scores[0], linestyle='--', color='red', label='OLS - Train')
        plt.plot(np.log10(lambdas), MSE_test_scores[0], linestyle='-', color='red', label='OLS - Test')

        plt.plot(np.log10(lambdas), MSE_train_scores[1], linestyle='--', color='blue', label='Ridge - Train')
        plt.plot(np.log10(lambdas), MSE_test_scores[1], linestyle='-', color='blue', label='Ridge - Test')

        plt.plot(np.log10(lambdas), MSE_train_scores[2], linestyle='--', color='orange', label='Lasso - Train')
        plt.plot(np.log10(lambdas), MSE_test_scores[2], linestyle='-', color='orange', label='Lasso - Test')
        
        plt.yscale('log')
        plt.legend()
        plt.xlabel(r'$log_{10}$($\lambda$)')
        plt.ylabel(r'MSE score')
        plt.grid()
        plt.tight_layout()
        save_fig(fig_name + "_N=" + str(N_data) + "_Noise=" + str(noise))
        plt.show()