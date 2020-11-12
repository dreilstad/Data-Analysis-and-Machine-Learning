import numpy as np
import tools as tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-dark')


class Layer():

    def __init__(self, inputs, neurons, activation="none", lmbda=0.01, weights=None, bias=None):

        self.inputs = inputs
        self.neurons = neurons
        self.activation = activation
        self.lmbda = lmbda

        self.error = None
        self.delta = None
        self.activation_result = None


        if weights is None:
            self.init_weights_and_bias()
        else:
            self.weights = weights
            self.bias = bias

        # is set when SGD variant ADAM or RMSprop is chosen
        self.s = None
        self.m = None


    def init_weights_and_bias(self):
        """Initializes the weights and biases for the layer. 
        
        The bias is randomly initialized.

        If the chosen activation function is
        either sigmoid or tanh, the Xavier initialization is used for the weights.

        If the chosen activation function is
        either relu or leakyrelu, the He initialization is used for the weights.

        xavier initialization:  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?source=post_page
        he initialization:      https://arxiv.org/pdf/1502.01852.pdf
        """

        # for sigmoid-type activation functions
        if self.activation == "sigmoid" or self.activation == "tanh":
            xavier_init = np.sqrt(6) / np.sqrt(self.inputs + self.neurons)
            self.weights = np.random.uniform(-xavier_init, xavier_init, size=(self.inputs, self.neurons))
        
        # for relu/leakyrelu-type activation functions
        elif self.activation == "relu" or self.activation == "leakyrelu":
            he_init = np.sqrt(2 / self.neurons)
            self.weights = np.random.uniform(size=(self.inputs, self.neurons)) * he_init
        
        else:
            self.weights = np.random.uniform(size=(self.inputs, self.neurons))
        
        self.bias = np.random.rand(self.neurons)

    def activate(self, x):
        """Applies the weights and bias to the input form the previous layer, and feeds the results 
        into an activation function. The result is saved and returned.

        Options for activation fucntion are:
            none (linear)
            sigmoid
            tanh
            relu
            leakyrelu
            softmax

        Args:
            x (ndarray): input form previous layer

        Returns:
            the result from applying the activation function
        """
        
        # activate layer
        z = (x @ self.weights) + self.bias

        if self.activation == "none":
            self.activation_result = z
            
        elif self.activation == "sigmoid":
            self.activation_result = 1/(1 + np.exp(-z))

            # clips the gradient to prevent it form "exploding"
            self.activation_result = np.clip(self.activation_result, -500, 500)

        elif self.activation == "tanh":
            self.activation_result = np.tanh(z)

        elif self.activation == "relu":
            z = np.where(z < 0, 0, z)
            self.activation_result = z

        elif self.activation == "leakyrelu":
            z = np.where(z < 0, 0.01*z, z)
            self.activation_result = z

        elif self.activation == "softmax":
            exp_term = np.exp(z - np.max(z))
            self.activation_result = exp_term / np.sum(exp_term)


        return self.activation_result

    def derivative(self, a):
        """Returns the derivative of the output from the activation function

        Args:
            a (ndarray): output from activation function

        Returns:
            the derivative of the output from th eactivation function

        """
        
        if self.activation == "none":
            return np.ones(a.shape)

        if self.activation == "sigmoid":
            return a * (1 - a)

        if self.activation == "tanh":
            return 1 - a**2

        if self.activation == "relu":
            a = np.where(a > 0, 1, 0)
            return a

        if self.activation == "leakyrelu":
            a = np.where(a < 0, 0.01, 1)
            return a
        
        return a

class FFNN():
    """
    Feed Forward Neural Network
    """

    def __init__(self, optimizer="standard"):
        self.layers = []
        self.optimizer = optimizer
        
        if optimizer == "RMSprop" or optimizer == "ADAM":
            self.j = 1

    def add_layer(self, layer):
        """Adds layer ot the neural network. If the network uses RMSprop or ADAM, we initialize variables for the layer used later.

        Args:
            layer (Layer): Layer object to add
        """
        self.layers.append(layer)

        if self.optimizer == "RMSprop":
            layer.s = np.zeros_like(layer.weights)

        elif self.optimizer == "ADAM":
            layer.m = np.zeros_like(layer.weights)
            layer.s = np.zeros_like(layer.weights)


    def compute_deltas(self, y, output):
        """Iterates the layers in the neural network backwards calculating the error and then the deltas for each layer.

        Args:
            y (ndarray): target values
            output (ndarray): output form a feed forward pass
        """
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # if layer is the output layer we use the ouput, 
            if layer == self.layers[-1]:
                layer.error = output - y
                layer.delta = layer.error * layer.derivative(output)

            # else we use the output from the activation function
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.derivative(layer.activation_result)

              
    def update_weights(self, X, learning_rate, lmbda):
        """Updates the weights and biases using the errors calculated earlier for each layer in the neural network. 

        Options for optimizers is standard SGD, RMSprop and ADAM.

        Args:
            X (ndarray): input data
            learning_rate (float): learning rate
            lmbda (float): regularization value
        """

        if self.optimizer == "standard":

            for i in range(len(self.layers)):
                layer = self.layers[i]
                
                # if layer is the first we use the input, else we use the activation result from the previous layer
                if i == 0:
                    X_l = np.atleast_2d(X)
                else:
                    X_l = np.atleast_2d(self.layers[i - 1].activation_result)
                
                # derivative w.r.t. the weights
                layer_gradient = layer.delta * X_l.T

                # update weights with or without regularization
                if lmbda > 0.0: 
                    layer.weights = (1 - learning_rate * lmbda) * layer.weights - layer.delta * X_l.T * learning_rate
                else:
                    layer.weights = layer.weights - layer.delta * X_l.T * learning_rate
                    
                # update the biases
                layer.bias = layer.bias - layer.delta * learning_rate 

        elif self.optimizer == "RMSprop":

            rho = 0.9
            eps = 1e-8
            
            for i in range(len(self.layers)):
                layer = self.layers[i]
                
                if i == 0:
                    X_l = np.atleast_2d(X)
                else:
                    X_l = np.atleast_2d(self.layers[i - 1].activation_result)

                layer_gradient = layer.delta * X_l.T
                
                if lmbda > 0.0: 
                    layer_gradient = layer_gradient + lmbda * layer.weights

                layer.s = (rho * layer.s) + ((1 - rho) * layer_gradient * layer_gradient)

                layer.weights = layer.weights - layer_gradient * learning_rate / np.sqrt(layer.s + eps)
                layer.bias = layer.bias - layer.delta * learning_rate 
        
        elif self.optimizer == "ADAM":

            rho_1 = 0.9
            rho_2 = 0.99
            eps = 1e-8
            
            for i in range(len(self.layers)):
                layer = self.layers[i]
                
                if i == 0:
                    X_l = np.atleast_2d(X)
                else:
                    X_l = np.atleast_2d(self.layers[i - 1].activation_result)

                layer_gradient = layer.delta * X_l.T
                
                if lmbda > 0.0: 
                    layer_gradient = layer_gradient + lmbda * layer.weights

                
                layer.m = (rho_1 * layer.m) + ((1 - rho_1) * layer_gradient)
                layer.s = (rho_2 * layer.s) + ((1 - rho_2) * layer_gradient * layer_gradient)

                m_hat = layer.m / (1 - rho_1 ** self.j)
                s_hat = layer.s / (1 - rho_2 ** self.j)

                layer.weights = layer.weights - learning_rate * (m_hat / (np.sqrt(s_hat) + eps))
                layer.bias = layer.bias - layer.delta * learning_rate 

            self.j += 1

    def feed_forward(self, X):  
        """Iterates the layers in the neural network and activates the neurons in the layers.

        Args:
            X (ndarray): input data

        Returns:
            the output from the last layer
        """  
        for layer in self.layers:
            X = layer.activate(X)

        self.probabilities = X
        return X

    def backpropogation(self, X, y, learning_rate, lmbda):
        """Performs the backpropagation pass by first doing a feed forward pass, the caclulate the errors/deltas,
        and then update the weights and biases.

        Args:
            X (ndarray): training sample
            y (ndarray): training sample
            learning_rate (float): learning rate
            lmbda (float): regularization value
        """

        # forward pass
        output = self.feed_forward(X)
        
        # compute deltas in each layer
        self.compute_deltas(y, output)
 
        # Update the weights
        self.update_weights(X, learning_rate, lmbda)
        


    def train(self, X, y, learning_rate, epochs, size_batch=1, lmbda=0.1):
        """Trains the neural network by performing the feed forward pass and backpropagation pass
        for a number of epochs using minibatches of the training set. 

        The training set is shuffled for each epoch.

        Args:
            X (ndarray): training sample
            y (ndarray): training sample
            learning_rate (float): learning rate
            epochs (int): number of epochs
            size_batch (int): size of the minibatches
            lmbda (float): regularization value
        """

        num_batches = X.shape[0] // size_batch

        indices = np.arange(X.shape[0])
        shuffler = np.random.default_rng()

        for epoch in range(epochs):
            
            # shuffles dataset
            shuffler.shuffle(indices)
            for i in range(num_batches):

                batch_indices = np.random.choice(indices, size_batch, replace=True)
                for j in batch_indices:
                    self.backpropogation(X[j], y[j], learning_rate, lmbda)


    def predict(self, X):
        """Performs a single feed forward pass which outputs a prediction. 
        
        If the neural network is applied to classification problem, 
        the class with the highest probability is returned.

        If the neural network is applied ot a regression problem, 
        the output is returned.

        Args:
            X (ndarray): test set to predict on

        Return:
            prediction
        """

        # feed forward pass
        output = self.feed_forward(X)

        # if last layers activation function is the softmax function, it is a classification problem
        if self.layers[-1].activation == "softmax":
            if output.ndim == 1:
                output = np.argmax(output)
            else:
                output = np.argmax(output, axis=1)

        return output

    def MSE(self, z, z_prediction):
        """Calculates the mean squared error using target values and the predicted values.

        Returns:
            MSE score
        """
        return (1 / len(z)) * np.sum((z_prediction - z) ** 2)

    def R2(self, z, z_prediction):
        """Calculates the R2-score using target values and the predicted values.

        Returns:
            R2 score
        """
        return 1 - (np.sum((z - z_prediction) ** 2)) / (np.sum((z - np.mean(z)) ** 2))

    def accuracy(self, target, prediction):
        """Calculates the accuracy score using target values and the predicted values.

        Returns:
            accuracy score
        """
        accuracy = float(sum(prediction == target) / len(target))
        return accuracy



