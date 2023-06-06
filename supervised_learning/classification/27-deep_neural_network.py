import numpy as np

def sigmoid(Z):
    """Sigmoid Activation Function"""
    return 1 / (1 + np.exp(-Z))

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """Deep Neural Network Constructor"""
        np.random.seed(0)
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {'W1': np.random.randn(layers[0], nx) * np.sqrt(2 / nx),
                          'b1': np.zeros((layers[0], 1))}
        for i in range(1, self.__L):
            self.__weights['W' + str(i + 1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    def forward_prop(self, X):
        """Forward Propagation"""
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            Z = np.matmul(W, self.__cache['A' + str(i - 1)]) + b
            A = sigmoid(Z)
            self.__cache['A' + str(i)] = A

        return A, self.__cache

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the neural network"""
        m = X.shape[1]
        cost_list = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            self.gradient_descent(Y, cache, alpha)

            if i % 1000 == 0:
                print('Cost after', i, 'iterations:', cost)
                cost_list.append(cost)

        return A, cost_list

    def cost(self, Y, A):
        """Compute the cost"""
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A))) / m
        return cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform gradient descent"""
        m = Y.shape[1]
        dZ = cache['A' + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.matmul(self.__weights['W' + str(i)].T, dZ) * (A_prev * (1 - A_prev))
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db


# Example usage
np.random.seed(5)
nx, m = np.random.randint(100, 200, 2).tolist()
classes = np.random.randint(5, 20)
X = np.random.randn(nx, m)
Y = np.random.randint(0, classes, m)

deep = DeepNeuralNetwork(nx, [100, 50, classes])
A, cost = deep.train(X, Y, iterations=10)
print(A)
print(cost)
