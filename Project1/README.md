# Project 1 - Regression analysis and resampling methods

## Report
Report directory contains the .pdf-file for the report and also the .tex-file.

## Results
Results directory contains all figures and results used in the report.

## Tests
Tests directory contains in several examples of commands to run with src, and the result you should expect.

In addition, the file **test_inversion.py** in the directory tests the code for the Singular Value Decomposition (SVD) algorithm with an equivalent inversion function **numpy.linalg.pinv** from Numpy.

## src
src directory contains all source code for the project.

- **analysis.py**
    - Contains code for model assessment, including MSE measurements and functions for plotting results.

- **resampling.py**
    - Contains code for the resmpling methods: the bootstrap method and k-fold cross validation.

- **tools.py**
    - Contains code for functions used for specific parts of the project.

- **ols.py**
    - Contains the class for Ordinary Least Squares (OLS).

- **ridge.py**
    - Contains the class for Ridge regression.

- **lasso.py**
    - Contains the class for Lasso regression.

- **project1.py**  
    Is the script which uses the rest of the .py-files. The script uses python library 'argsparse' to interpret arguments from the terminal
    
    To use the script use the command
    <pre><code> $ python3 project1.py -arguments & flags- </code></pre>

    The command 
    <pre><code> $ python3 project1.py -h </code></pre>
    shows available arguments

    Check the code in **project1.py** to see which arguments is needed for the different functions. The arguments needed depends on the function chosen.

    Example(the order of the flags does not matter):
    <pre><code> $ python3 project1.py -d franke -N 1000 -no 0.04 -m OLS -f biasVariance -deg 15 -b 50 </code></pre>
    The result is a plot of the **bias-variance** decomposition using **OLS** on the **Franke** function with **N datapoints = 1000**, **noise = 0.04**, **polynomial degrees = 1 - 15** og **bootstraps = 50**.

## Packages required to use the code

**Numpy**
- Install using: **pip3 install numpy**
- Earliest version working with code: 1.18.1 >=

**Scikit-Learn**
- Install using: **pip3 install scikit-learn**
- Earliest version working with code: 0.23.2 >=

**Seaborn**
- Install using: **pip3 install seaborn**
- Earliest version working with code: 0.10.1 >=

**Imageio**
- Install using: **pip3 install imageio**
- Earliest version working with code: 2.6.1 >=