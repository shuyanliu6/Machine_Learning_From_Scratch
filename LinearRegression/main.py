import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.0001, epochs=300):
        """
        Initialize the LinearRegression model with learning rate and epochs.
        
        Args:
            learning_rate (float): The rate at which the model learns. Default is 0.0001.
            epochs (int): The number of iterations to perform for gradient descent. Default is 300.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.b0 = 0  # Intercept
        self.b1 = 0  # Slope
    
    def gradient_descent(self, x, y):
        """
        Perform one iteration of gradient descent to update the model parameters (b0 and b1).
        
        Args:
            x (array-like): Independent variable (feature).
            y (array-like): Dependent variable (target).
        
        Returns:
            None: Updates the model's b0 and b1 attributes in place.
        """
        b0_gradient = 0
        b1_gradient = 0
        n = len(x)
        
        for i in range(n):
            # Calculate gradients
            b0_gradient += -(2/n) * (y[i] - (self.b0 + self.b1 * x[i]))
            b1_gradient += -(2/n) * (y[i] - (self.b0 + self.b1 * x[i])) * x[i]

        
        # Update parameters
        self.b0 -= self.learning_rate * b0_gradient
        self.b1 -= self.learning_rate * b1_gradient

    def fit(self, x, y):
        """
        Train the linear regression model using gradient descent.
        
        Args:
            x (array-like): Independent variable (feature), a 1D array of input data.
            y (array-like): Dependent variable (target), a 1D array of corresponding output data.
        
        Returns:
            self: Returns the instance itself for method chaining.
        """
        x = np.array(x)
        y = np.array(y)
        
        for _ in range(self.epochs):
            self.gradient_descent(x, y)
        
        return self

    def predict(self, x):
        """
        Predict the dependent variable (y) using the trained model parameters.
        
        Args:
            x (array-like): Independent variable (feature) values to make predictions on.
        
        Returns:
            np.ndarray: Predicted values of the dependent variable (y) for the input feature (x).
        """
        x = np.array(x)
        return self.b0 + self.b1 * x

    def get_params(self):
        """
        Get the learned parameters b0 (intercept) and b1 (slope).
        
        Returns:
            tuple: The intercept (b0) and slope (b1) of the model.
        """
        return self.b0, self.b1

    ## def model_valuate
    ## mutivariate 
    