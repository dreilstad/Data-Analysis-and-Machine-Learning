import sys
import numpy as np

from regression import Regression
from ridge import Ridge
from lasso import Lasso
from analysis import Analysis
from resampling import Bootstrap, CrossValidation
from tools import generateData, frankeFunction, computeDesignMatrix


part = sys.argv[1]
N = int(sys.argv[2])
noisy = float(sys.argv[3])
poly_degree = int(sys.argv[4])


if part == 'a':
    partA(N, noisy, poly_degree)

elif part == 'b':
    bootstraps = int(sys.argv[5])
    partB(N, noisy, poly_degree, bootstraps, True)

elif part == 'c':
    kfolds = int(sys.argv[5])
    partC(N, noisy, poly_degree, kfolds, compare=True)

elif part == 'd':
    bootstraps = int(sys.argv[5])
    N_lambdas = int(sys.argv[6])
    partD(N, noisy, poly_degree, bootstraps, N_lambdas, 'ridge_bias_variance')


