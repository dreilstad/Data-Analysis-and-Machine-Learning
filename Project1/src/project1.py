import sys
import numpy as np
import tools
import argparse

from analysis import Analysis

terrain_file = 'SRTM_data_Norway_2.tif'

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Project 1 - How to use script', add_help=False)
    parser._action_groups.pop()
    possible_args = parser.add_argument_group('possible arguments')

    possible_args.add_argument('-d', '--data', 
                               type=str, 
                               required=True,
                               choices=['terrain', 'franke'],
                               help='Choose terrain data or Franke function')

    possible_args.add_argument('-m', '--method', 
                               type=str, 
                               required=False,
                               choices=['OLS', 'Ridge', 'Lasso'],
                               help='Choose regression method')
                               
    possible_args.add_argument('-f', '--function', 
                               type=str, 
                               required=True,
                               choices=['compareAll', 'biasVariance', 'crossValidation', 'errorVsComplexity', 'errorVsComplexityNoResampling', 'lambdaVsComplexity'],
                               help='Choose which function to use')

    possible_args.add_argument('-deg', '--degree', 
                               type=int, 
                               required=True,
                               help='Specify max polynomial degree')


    possible_args.add_argument('-N', '--N_datapoints', 
                               type=int, 
                               required=False,
                               help='Specify number of datapoints when Franke function is chosen as datatype')

    possible_args.add_argument('-no', '--noise', 
                               type=float, 
                               required=False,
                               default=0.0,
                               help='Specify noise when Franke function is chosen as datatype')

    possible_args.add_argument('-l', '--lmbda', 
                               type=float, 
                               required=False,
                               default=0.0,
                               help='Specify lambda value when using Ridge or Lasso')

    possible_args.add_argument('-b', '--bootstrap', 
                               type=int, 
                               required=False,
                               default=50,
                               help='Specify number of bootstraps when resampling')

    possible_args.add_argument('-k', '--kfolds', 
                               type=int, 
                               required=False,
                               default=10,
                               help='Specify number of folds when resampling') 

    possible_args.add_argument('-re', '--resampling', 
                               type=str, 
                               required=False,
                               choices=['cv', 'boot'],
                               default='cv',
                               help='Specify resampling technique')

    possible_args.add_argument('-ci', '--confidence_interval', 
                               type=int,
                               required=False,
                               default=-1,
                               help='Choose for which degree to show confidence interval of beta')

    possible_args.add_argument('-r2', '--r2_score', 
                               action='store_true',
                               required=False,
                               help='Add flag -r2 with -f "errorVsComplexityNoResampling" to show R2-scores')

    possible_args.add_argument('-c', '--compare', 
                               action='store_true',
                               required=False,
                               help='Add flag -c with -f "crossValidation" to compare with bootstrap')

    possible_args.add_argument('-h', '--help',
                               action='help',
                               help='Helpful message showing flags and usage of instapy')

    args = parser.parse_args()


    datatype = args.data
    N = args.N_datapoints
    noise = args.noise

    if datatype == 'terrain':
        terrain_data = tools.readTerrain(terrain_file, show=False)
        terrain = terrain_data[300:400, 400:500]

        x = np.linspace(0,1, terrain.shape[0])
        y = np.linspace(0,1, terrain.shape[1])
        x, y = np.meshgrid(x, y)

        # flatten to 1D
        x = x.flatten()
        y = y.flatten()
        z = terrain.flatten()

        # normalize
        z = z - np.min(z)
        z = z / np.max(z)

    elif datatype == 'franke':
        x, y = tools.generateData(N)
        z = tools.frankeFunction(x, y, noise=noise)

    
    function = args.function
    method = args.method
    degrees = args.degree
    lmbda = args.lmbda
    kfolds = args.kfolds
    bootstraps = args.bootstrap
    technique = args.resampling
    ci = args.confidence_interval
    r2 = args.r2_score
    compare = args.compare


    if function == 'compareAll':
        tools.compareAll(x, y, z, noise, degrees, lmbda, kfolds)

    elif function == 'biasVariance':
        tools.biasVariance(x, y, z, noise, degrees, method, lmbda, bootstraps)

    elif function == 'crossValidation':
        tools.crossValidation(x, y, z, noise, degrees, method, lmbda, bootstraps=bootstraps, kfolds=kfolds, compare=compare)

    elif function == 'errorVsComplexity':
        if technique == 'cv':
            tools.errorVsComplexity(x, y, z, noise, degrees, method, lmbda, kfolds, technique=technique)
        else:
            tools.errorVsComplexity(x, y, z, noise, degrees, method, lmbda, bootstraps, technique=technique)

    elif function == 'errorVsComplexityNoResampling':
        tools.errorVsComplexityNoResampling(x, y, z, noise, degrees, method, lmbda, R2=r2, CI=ci)

    elif function == 'lambdaVsComplexity':
        tools.lambdaVsComplexity(x, y, z, noise, degrees, method, kfolds)


    