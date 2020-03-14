import numpy as np

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
        """
            X: array-like, shape = [n_samples, n_features], Training vectors
            y: array-like, shape = [n_samples], Target values 
        """
        rgen = np.random.RandomState(self.random_state)
        """
            rgen: NumPy random number generator
            np.random.RandomState: A method for generating random numbers
        """
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        """
            _: An attribute that won't be created upon initialization of the object, 
               but by calling other methods. In Self.W_, weights will be initialized 
               to a vector R^m=1 where m: features.
           rgen.normal(...): Random numbers drawn from a normal distribution with standard
                             deviation 0.01, Mean of distribution 0.0, Output shape 1
        """
        self. errors_ = []
        # Collects the number of misclassifications during each epoch
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                # for each value in the training data and target data
                update = self.eta * (target - self.predict(xi))
                # Run the learning function
                self.w_[1:] += update * xi
                self.w_[0] += update
                # Adds the scaled error to the weight vector
                errors += int(update != 0.0)
                # Updates the error counter
            self.errors_.append(errors)
            # Records the number of errors in the error list
        return self
        # Return the object
    
    def net_input(self, X):
        # Calculates the net input
        return np.dot(X, self.w_[1:]) + self.w_[0]
        """
            np.dot: A function that calculates the vector dot product and returns 
            the dot product of a training data entry with the weights
        """

    def predict(self, X):
        # Return class label after unit step
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        """
            Returns a prediction of -1 or 1 depending on a positive or negative 
            dot product
        """
        
