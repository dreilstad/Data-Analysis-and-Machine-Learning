import numpy as np
import analysis as an
class NeuralNetwork():

    def __init__(self, 
                 X_full, 
                 y_full, 
                 learning_rate=0.01, 
                 epochs=100, 
                 size_batch=10, 
                 n_hidden_neurons_per_layer=[50],
                 n_hidden_layers=1, 
                 n_categories=10, 
                 activation="sigmoid",
                 activation_output="linear",
                 sgd_variant="standard",
                 problem_type="regression", # regression or classification
                 lmbda=0.0):

        self.X_full = X_full
        self.y_full = y_full

        N,P = X_full.shape
        self.n_inputs = N
        self.n_features = P

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.size_batch = size_batch
        self.num_batches = int(N / size_batch)

        self.n_hidden_neurons_per_layer = n_hidden_neurons_per_layer
        self.n_hidden_layers = n_hidden_layers
        self.n_categories = n_categories

        self.activation = activation
        self.activation_output = activation_output
        self.sgd_variant = sgd_variant
        self.problem_type = problem_type
        self.lmbda = lmbda

        self.layers = []

        self.initialize_layers()


    def initialize_layers(self):


        first_layer = Layer(self.n_features, self.n_hidden_neurons_per_layer[0], self.activation)
        self.layers.append(first_layer)

        for i in range(1, self.n_hidden_layers):
            layer = Layer(self.n_hidden_neurons_per_layer[i - 1], self.n_hidden_neurons_per_layer[i], self.activation)
            self.layers.append(layer)
        
        output_layer = Layer(self.n_hidden_neurons_per_layer[-1], self.n_categories, None)
        self.layers.append(output_layer)


    def update(self):
        pass


    def feed_forward(self, X):

        for layer in self.layers:
            X = layer.activate(X)

        self.probabilities = X
        return X

    def backpropogation(self):
        """
        http://neuralnetworksanddeeplearning.com/chap2.html
        """

        output_layer = self.layers[-1]
        if self.problem_type == "regression":
            error = an.MSE(self.y, self.probabilities)
        elif self.problem_type == "classification":
            error = self.y - self.probabilities

        output_layer.delta = error * output_layer.activate_derivative(self.probabilities)

        # compute error
        for i in range(len(self.n_hidden_layers) - 1, -1, -1):
            layer = self.layers[i]
            prev_layer = self.layers[i + 1]

            error = prev_layer.weights @ prev_layer.delta
            layer.delta = error * layer.activate_derivative(layer.activation_result)

        first_layer = self.layers[0]
        first_layer.weights = first_layer.weights + self.learning_rate * first_layer.delta * self.X.T

        # update weights and biases
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]

            # w_l = w_l + learning_rate * delta_l * a_(l - 1).T
            layer.weights = layer.weights - self.learning_rate * layer.delta * prev_layer.activation_result.T 


            #regularization = 1 - self.lmbda * self.learning_rate
            #layer.weights = layer.weights * regularization - self.learning_rate * layer.delta * prev_layer.activation_result.T 

            layer.biases = layer.biases - self.learning_rate * layer.delta 


    def train(self):

        for epoch in range(self.epochs):
            for i in range(self.num_batches):
                batch_indices = np.random.randint(self.N, size=self.size_batch)

                self.X = self.X_full[batch_indices]
                self.y = self.y_full[batch_indices]

                self.feed_forward(self.X)
                self.backpropogation()

    def predict(self, X_test):
        prediction = self.feed_forward(X_test)

        if self.problem_type == "regression":
            return prediction

        elif self.problem_type == "classification":
            best_class = np.argmax(prediction)
            certainty = prediction[best_class]
            return best_class, certainty

    def update(self):

        if self.sgd_variant == "standard":
            pass

        
class Layer():

    def __init__(self, inputs, neurons, activation=None, lmbda=0.0, alpha=0.01):

        self.inputs = inputs
        self.neurons = neurons
        self.activation = activation
        self.lmbda = lmbda
        self.alpha = alpha

        self.delta = None
        self.activation_result = None

        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):

        self.weights = np.random.randn(self.inputs, self.neurons)
        self.biases = np.zeros(self.neurons) + 0.01

    def activate(self, X):
        
        z = X @ self.weights + self.biases

        if self.activation is None:
            self.activation_result = z
            
        elif self.activation == "sigmoid":
            self.activation_result = 1.0/(1.0 + np.exp(-z))

        elif self.activation == "relu":
            pass

        elif self.activation == "leaky_relu":
            pass

        elif self.activation == "softmax":
            pass


        return self.activation_result

    def activate_derivative(self, z):

        if function is None:
            return z

        if function == "sigmoid":
            return z * (1 - z)