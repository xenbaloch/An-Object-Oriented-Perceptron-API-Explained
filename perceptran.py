import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron():
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
            eta: learning rate typically between 0.0 & 1.0
            n_iter: number of epocs set to 50
            random_state: int, random number generator seed for random weights
        """
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
        """_: An attribute that won't be created upon initialization of the object, 
               but by calling other methods. In Self.W_, weights will be initialized 
               to a vector R^m=1 where m: features.
           rgen.normal(...): Random numbers drawn from a normal distribution with standard
                             deviation 0.01, Mean of distribution 0.0, Output shape 1"""
        
        # Collects the number of misclassifications during each epoch
        self. errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            # for each value in the training data and target data
            for xi, target in zip(X,y):
                # Run the learning function
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # Adds the scaled error to the weight vector
                self.w_[0] += update
                # Updates the error counter
                errors += int(update != 0.0)
            # Records the number of errors in the error list
            self.errors_.append(errors)
            
        return self
    
    # Calculates the net input
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        """np.dot: A function that calculates the vector dot product and returns 
            the dot product of a training data entry with the weights"""
    # Return class label after unit step
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        """Returns a prediction of -1 or 1 depending on a positive or negative 
           dot product"""
        
v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
# np.arccos(): Calculates inverse cosine 
# np.linalg.norm(): A function that computes length of a vector
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# Reading-in the Iris data using pd.read_csv()
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)

# A function that checks wheather the data loads correctly
df.tail()

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-Setosa', -1, 1)
"""Extracted 100 class labels of 50 each, converted the class labels to two
   integer class labels 1(versicolor) and -1(setosa), assigned to vector y"""

X = df.iloc[0:100, [0, 2]].values
"""Extracted sepal length and petal length of 100 training samples and assigned
   them to a feature matrix X"""

plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# Training the Perceptron Model
ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# A function for plotting decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
        
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()
