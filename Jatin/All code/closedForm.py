import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2024)

class LinearRegressionClosedForm:
  def __init__(self):
    '''
    Initializing the parameters of the model

    Returns:
      None
    '''
    self.weights = None

  def fit(self, X, y):
    '''
    This function is used to obtain the weights of the model using closed form solution.

    Args:
      X : 2D numpy array of training set data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values in the training dataset. Dimensions (n x 1)

    Returns :
      None
    '''
    # Calculate the weights
    X_transpose = np.transpose(X)
    X_transpose_X = np.matmul(X_transpose, X)
    X_transpose_X_inv = np.linalg.inv(X_transpose_X)
    X_transpose_y = np.matmul(X_transpose, y)
    self.weights = np.matmul(X_transpose_X_inv, X_transpose_y)
    print("self.weights:")
    print(self.weights)
    # raise NotImplementedError()

  def predict(self, X):
    '''
    This function is used to predict the target values for the given set of feature values

    Args:
      X: 2D numpy array of data points. Dimensions (n x (d+1))

    Returns:
      2D numpy array of predicted target values. Dimensions (n x 1)
    '''
    # Write your code here
     # Check if weights are initialized
    if self.weights is None:
        raise ValueError("Model weights are not initialized. Call 'fit' before 'predict'.")

    # Compute the predicted values
    y_hat = np.matmul(X, self.weights)
    return y_hat
    # raise NotImplementedError()

def plot_learned_equation(X, y, y_hat):
    '''
    This function generates the plot to visualize how well the learned linear equation fits the dataset  

    Args:
      X : 2D numpy array of data points. Dimensions (n x 2)
      y : 2D numpy array of target values. Dimensions (n x 1)
      y_hat : 2D numpy array of predicted values. Dimensions (n x 1)

    Returns:
      None
    '''
    # Plot a 2d plot, with only  X[:,1] on x-axis (Think about why you can ignore X[:, 0])
    # Use y_hat to plot the line. DO NOT use y. 
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of the true data points (using feature values on x-axis and target values on y-axis)
    plt.scatter(X[:, 1], y, color='blue', label='True Data Points')

    # Plot the learned linear equation (y_hat) using feature values on x-axis
    plt.plot(X[:, 1], y_hat, color='red', label='Fitted Line')

    plt.xlabel('Feature Value')
    plt.ylabel('Target Value')
    plt.title('Predicted vs. True Data')
    plt.legend()
    plt.grid(True)
    plt.savefig("closed_form.png")

    plt.show()
    plt.close()
    
    # raise NotImplementedError()

############################################
#####        Helper functions          #####
############################################
def plot_data(X, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1], y, color='blue', label='Data Points')
    plt.xlabel('Feature Value')
    plt.ylabel('Target Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_toy_dataset():
    '''
    This function generates a simple toy dataset containing 300 points with 1d feature 
    '''
    # Generate random input feature matrix
    X = np.random.rand(300, 2)
    # print("X (initial random values):")
    # print(X)
    # plot_data(X, np.zeros((300, 1)), "X (initial random values)")

    # Set the first column to 1 to include the bias term
    X[:, 0] = 1
    # print("\nX (with bias term added):")
    # print(X)
    # plot_data(X, np.zeros((300, 1)), "X (with bias term added)")

    # Initialize weights randomly
    weights = np.random.rand(2,1)
    print("\nWeights:")
    print(weights)

    # Generate random noise
    noise = np.random.rand(300,1) / 32
    # print("\nNoise:")
    # print(noise)

    # Compute the target values using the feature matrix, weights, and adding noise
    y = np.matmul(X, weights) + noise
    # print("\nTarget values (y):")
    # print(y)
    # plot_data(X, y, "Target values (y)")
    
    X_train = X[:250]
    X_test = X[250:]
    y_train = y[:250]
    y_test = y[250:]
    return X_train, y_train, X_test, y_test

# Terminal text coloring
RESET = '\033[0m'
GREEN = '\033[32m'
RED = '\033[31m'

if __name__ == '__main__':
    
    print(RED + "##### Closed form solution for linear regression #####")
    
    print(RESET +  "Loading dataset: ",end="")
    try:
        X_train, y_train, X_test, y_test = generate_toy_dataset()
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Calculating closed form solution: ", end="")
    try:
        linear_reg = LinearRegressionClosedForm()
        linear_reg.fit(X_train,y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Predicting for test split: ", end="")
    try:
        y_hat = linear_reg.predict(X_test)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Plotting the solution: ", end="")
    try:
        plot_learned_equation(X_test, y_test, y_hat)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    