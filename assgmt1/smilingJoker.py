
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2024)
degree=2
# method = 'Gaussian'
# method ='polynomial'
method ='polynomial_sin_cos'
from closedForm import LinearRegressionClosedForm

def transform_input(x):
    '''
    This function transforms the input to generate new features.

    Args:
      x: 2D numpy array of input values. Dimensions (n' x 1)

    Returns:
      2D numpy array of transformed input. Dimensions (n' x K+1)
      
    '''
    
    centers=None
    gamma=1.0
    n = x.shape[0]
    X_transformed = np.ones((n, degree + 1))
    if method == 'polynomial':
    ############ for polynomial################
        # Generate polynomial features
        for d in range(1, degree + 1):
            X_transformed[:, d] = x[:, 0] ** d
            
        # Plotting the original feature vs. each polynomial feature
        plt.figure(figsize=(10, 6))

    # Plotting the original feature (X) vs. the first-degree polynomial feature
        plt.plot(x[:, 0], X_transformed[:, 1], 'o-', label='Original feature (degree 1)')

    # Plotting the original feature (X) vs. the second-degree polynomial feature
        plt.plot(x[:, 0], X_transformed[:, 2], 'o-', label='Squared feature (degree 2)')

        plt.title('Polynomial Features of X')
        plt.xlabel('Original Feature (X)')
        plt.ylabel('Transformed Features')
        plt.legend()
        plt.grid(True)
        plt.show()
    ################for gaussian bases#################
      # If centers are not provided, we can use some equally spaced points
    elif method == 'Gaussian':
        if centers is None:
            centers = np.linspace(np.min(x), np.max(x), degree)
        
        # Initialize the transformed X with the bias (ones)
        X_transformed = np.ones((n, len(centers) + 1))
        
        # Generate RBF features
        for i, c in enumerate(centers):
            X_transformed[:, i + 1] = np.exp(-gamma * (x[:, 0] - c) ** 2)
            
        # Plotting the original feature vs. each RBF-transformed feature
        plt.figure(figsize=(10, 6))
        
        for i, c in enumerate(centers):
            plt.plot(x[:, 0], X_transformed[:, i + 1], 'o-', label=f'RBF feature (center={c:.2f})')

        plt.title('RBF Features of X')
        plt.xlabel('Original Feature (X)')
        plt.ylabel('Transformed Features')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    elif method == 'polynomial_sin_cos':
        # Initialize the transformed X with the correct size
        # The size is (n, degree + 2 * degree), where n is the number of samples.
        X_transformed = np.ones((n, degree + 2) )
        # X_transformed = np.ones((n, degree + 2 * degree))
        X_transformed_df = pd.DataFrame(X_transformed)
        print('before')
        print(X_transformed_df)
        # Generate polynomial features
        for d in range(1, degree + 1):
            X_transformed[:, d - 1] = x[:, 0] ** d  # x^1, x^2, ..., x^degree
        X_transformed_df = pd.DataFrame(X_transformed)
        print('after')
        print(X_transformed_df)
        
        # print(X_transformed)
        # Generate sine and cosine features
        
        for d in range(1, degree ):
            sin_index = degree + 2 * (d - 1)
            cos_index = sin_index + 1
            X_transformed[:, sin_index] = np.sin(d * x[:, 0])  # sin(d * x)
            X_transformed[:, cos_index] = np.cos(d * x[:, 0])  # cos(d * x)

        # Plotting the original feature vs. polynomial, sine, and cosine transformed features
        plt.figure(figsize=(10, 6))
        
        # Plot polynomial features
        for d in range(1, degree + 1):
            plt.plot(x[:, 0], X_transformed[:, d - 1], 'o-', label=f'Polynomial feature (degree {d})')
        
        # Plot sine and cosine features
        for d in range(1, degree ):
            sin_index = degree + 2 * (d - 1)
            cos_index = sin_index + 1
            plt.plot(x[:, 0], X_transformed[:, sin_index], 'o-', label=f'Sine feature (harmonic {d})')
            plt.plot(x[:, 0], X_transformed[:, cos_index], 'o-', label=f'Cosine feature (harmonic {d})')
        
        # Show the plot
        plt.title('Polynomial, Sine, and Cosine Features of X')
        plt.xlabel('Original Feature (X)')
        plt.ylabel('Transformed Features')
        plt.legend()
        plt.grid(True)
        plt.show()
       
    return X_transformed
    # Write your code here
    # raise NotImplementedError()
    
def read_dataset(filepath):
    '''
    This function reads the dataset and creates train and test splits.
    
    n = 500
    n' = 0.9*n

    Args:
      filename: string containing the path of the csv file

    Returns:
      X_train: 2D numpy array of input values for training. Dimensions (n' x 1)
      y_train: 2D numpy array of target values for training. Dimensions (n' x 1)
      
      X_test: 2D numpy array of input values for testing. Dimensions ((n-n') x 1)
      y_test: 2D numpy array of target values for testing. Dimensions ((n-n') x 1)
      
    '''
    # Write your code here
       # Read the dataset from the CSV file
    data = pd.read_csv('dataset.csv')
    
    # Assume the CSV file has columns 'x' and 'y' for features and target values respectively
    X = data[['x']].values
    y = data[['y']].values
    
    # Determine the split index
    n = X.shape[0]
    split_index = int(0.9 * n)
    
    # Split the data into training and testing sets
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    
    return X_train, y_train, X_test, y_test
    # raise NotImplementedError()


############################################
#####        Helper functions          #####
############################################

def plot_dataset(X, y):
    '''
    This function generates the plot to visualize the dataset  

    Args:
      X : 2D numpy array of data points. Dimensions (n x 1)
      y : 2D numpy array of target values. Dimensions (n x 1)

    Returns:
      None
    '''
    plt.title('Plot of the unknown dataset')
    plt.scatter(X, y, color='r')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.savefig('dataset.png')

# Terminal text coloring
RESET = '\033[0m'
GREEN = '\033[32m'
RED = '\033[31m'

if __name__ == '__main__':
    
    print(RED + "##### Starting experiment #####")
    
    print(RESET +  "Loading dataset: ",end="")
    try:
        X_train, y_train, X_test, y_test = read_dataset('dataset.csv')
        print(GREEN + "done")
        original_x_test=np.copy(X_test)
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET +  "Plotting dataset: ",end="")
    try:
        plot_dataset(X_train, y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Performing input transformation: ", end="")
    try:
        X_train = transform_input(X_train)
        X_test = transform_input(X_test)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
        
    print(RESET + "Caclulating weights: ", end="")
    try:
        linear_reg = LinearRegressionClosedForm()
        linear_reg.fit(X_train,y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Checking closeness: ", end="")
    try:
        y_hat = linear_reg.predict(X_test)
        plt.title('predicted vs original input')
        # plt.scatter(original_x_test, y_hat, color='r')
        plt.scatter(original_x_test, y_test, color='blue', label='True Data Points')

    # Plot the learned linear equation (y_hat) using feature values on x-axis
        plt.plot(original_x_test, y_hat, color='red', label='Fitted Line')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.legend()
        plt.grid(True)
        if method == 'polynomial':
            plt.savefig(f"actual_vs_predicted_for_{degree}_degree_polynomial.png")
        elif method == 'Gaussian':
            plt.savefig("actual_vs_predicted_forgaussian bases.png")
        plt.show()
        plt.close()
        # plt.savefig('dataset_predicted.png')
        
        print(np.allclose(y_hat, y_test, atol=1e-02))
    #    print(y_hat-y_test)
        # if np.allclose(y_hat, y_test, atol=1e-02):
        if np.allclose(y_hat, y_test, atol=1e-02):
          print(GREEN + "done")
        else:
          print(RED + "failed")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()