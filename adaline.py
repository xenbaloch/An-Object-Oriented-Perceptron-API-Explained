import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class AdalineGD():
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """eta: learning rate typically between 0.0 & 1.0
            n_iter: number of epocs set to 50
            random_state: int, random number generator seed for random weights"""
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """X: array-like, shape = [n_samples, n_features], Training vectors
            y: array-like, shape = [n_samples], Target values"""
        rgen = np.random.RandomState(self.random_state)
        """rgen: NumPy random number generator
            np.random.RandomState: A method for generating random numbers"""
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        """ _ : An attribute that won't be created upon initialization of the object, 
               but by calling other methods. In Self.W_, weights will be initialized 
               to a vector R^m=1 where m: features.
           rgen.normal(...): Random numbers drawn from a normal distribution with standard
                             deviation 0.01, Mean of distribution 0.0, Output shape 1"""
        
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            
        return self
    
    def net_input(self, X):
        # Calculates the net input
        return np.dot(X, self.w_[1:]) + self.w_[0]
        """np.dot: A function that calculates the vector dot product and returns 
            the dot product of a training data entry with the weights"""
        
    def activation(self, X):
        return X

    def predict(self, X):
        # Return class label after unit step
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        """Returns a prediction of -1 or 1 depending on a positive or negative 
            dot product"""
        
v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
df.tail()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()
