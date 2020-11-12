# Project 2 - Classification and Regression, from linear and logistic regression to neural networks


## Report
Report directory contains the .pdf-file for the report adn also the .tex-file.

## Results
Results contains all selected figures and results used in the report.

## Tests
Tests directory contains several examples of commands to run with the files in the src directory, and also results you should expect.

## src
src directory contains all source code used to produce the results in the report.

- **analysis.py**
    - Contains code for measuring the quality of the models, and also functions used to plot the results.

- **resampling.py**
    - Contains code for the resampling method k-fold cross validation. The code is reused from project 1.

- **tools.py**
    - Contains code for functions used for specific parts of the project.

- **ols.py**
    - Contains the class for Ordinary Least Squares (OLS).

- **ridge.py**
    - Contains the class for Ridge regression.

- **sgd.py**
    - Contains the class for Stochastic Gradient Descent.

- **neural_network.py**
    - Contains the class for Feed Forward Neural Network and the Layer class used in the neural network.  

        Example code for creating a neural network:
        <pre><code>  ffnn = FFNN(optimizer=optimizer)
        ffnn.add_layer(Layer(n_inputs, 10, "relu"))
        ffnn.add_layer(Layer(10, 5, "relu"))
        ffnn.add_layer(Layer(5, n_categories, "softmax")) </code></pre>

        The example code above creates a Feed Forward Neural Network with 2 hidden layers and 1 output layer. The first hidden layer contains 10 neurons, and the second hidden layer contains 5 neurons. The RELU function is specified as activation functin in the hidden layer, while softmax is used int he output layer.

- **logistic.py**
    - Contains the class for multinomial logistic regression.

Runnable files:

- **partA.py**  
    Contains the code used to generate the results in the report for part a).

    To use the script use the command
    <pre><code> $ python3 partA.py -arguments & flags- </code></pre>

    The command 
    <pre><code> $ python3 partA.py -h </code></pre>
    shows available arguments

    Check the code in **partA.py** to see which arguments is needed for the different functions. The arguments needed depends on the function chosen.

- **partBC.py**  
    Contains the code used to generate the results in the report for part b) and c).

    To use the script use the command
    <pre><code> $ python3 partBC.py -arguments & flags- </code></pre>

    The command 
    <pre><code> $ python3 partBC.py -h </code></pre>
    shows available arguments

    Check the code in **partBC.py** to see which arguments is needed for the different functions. The arguments needed depends on the function chosen.

- **partDE.py**  
    Contains the code used to generate the results in the report for part d) and e).

    To use the script use the command
    <pre><code> $ python3 partDE.py -arguments & flags- </code></pre>

    The command 
    <pre><code> $ python3 partDE.py -h </code></pre>
    shows available arguments

    Check the code in **partDE.py** to see which arguments is needed for the different functions. The arguments needed depends on the function chosen.

## Packages required to use the code

**Numpy**
- Install using: **pip3 install numpy**

**Scikit-Learn**
- Install using: **pip3 install scikit-learn**

**Matplotlib**
- Install using: **pip2 install matplotlib**

**Seaborn**
- Install using: **pip3 install seaborn**