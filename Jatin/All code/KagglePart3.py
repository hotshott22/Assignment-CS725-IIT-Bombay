import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2024)

class LinearRegressionBatchGD:
    def __init__(self, learning_rate=0.01, max_epochs=100, batch_size=1000):
        '''
        Initializing the parameters of the model

        Args:
          learning_rate : learning rate for batch gradient descent
          max_epochs : maximum number of epochs that the batch gradient descent algorithm will run for
          batch_size : size of the batches used for batch gradient descent.

        Returns:
          None
        '''
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weights = None

    def fit(self, X, y, plot=True):
        '''
        This function is used to train the model using batch gradient descent.

        Args:
          X : 2D numpy array of training set data points. Dimensions (n x (d+1))
          y : 2D numpy array of target values in the training dataset. Dimensions (n x 1)

        Returns :
          None
        '''
        if self.batch_size is None:
            self.batch_size = X.shape[0]  # Set batch size to the number of data points if not provided

        # Initialize the weights
        self.weights = np.zeros((X.shape[1], 1))
        print("Initial weights:")
        print(self.weights)
        prev_weights = np.copy(self.weights)
        prev_loss=0.000000000
        self.error_list = []  # Stores the loss for every epoch

        for epoch in range(self.max_epochs):
            
            batches = create_batches(X, y, self.batch_size)
            for X_batch, y_batch in batches:
                dw = self.compute_gradient(X_batch, y_batch, self.weights)

                self.weights -= self.learning_rate * dw

            loss = self.compute_rmse_loss(X, y, self.weights)
            self.error_list.append(loss)

            
            print("epoch:" )
            print(epoch)
            print("Loss:")
            print(loss)
            if np.linalg.norm(self.weights - prev_weights) < 1e-2:
                print("Convergence achieved.")
                break
            elif np.linalg.norm(loss - prev_loss) < 1e-5:
              print("loss is minimum.")
              break
            prev_weights = np.copy(self.weights)
            prev_loss=np.copy(loss)
        if plot:
            plot_loss(self.error_list, epoch + 1)

    def predict(self, X):
        '''
        This function is used to predict the target values for the given set of feature values

        Args:
          X: 2D numpy array of data points. Dimensions (n x (d+1))

        Returns:
          2D numpy array of predicted target values. Dimensions (n x 1)
        '''
        
        
        predictions = np.dot(X, self.weights)
        return predictions

    def compute_rmse_loss(self, X, y, weights):
        '''
        This function computes the Root Mean Square Error (RMSE)

        Args:
          X : 2D numpy array of data points. Dimensions (n x (d+1))
          y : 2D numpy array of target values. Dimensions (n x 1)
          weights : 2D numpy array of weights of the model. Dimensions ((d+1) x 1)
        
        
        '''
        
        predictions = np.dot(X, weights)        
        mse = np.mean((predictions - y) ** 2)        
        rmse = np.sqrt(mse)
        return rmse

    def compute_gradient(self, X, y, weights):
        '''
        This function computes the gradient of mean squared-error loss w.r.t the weights

        Args:
          X : 2D numpy array of data points. Dimensions (n x (d+1))
          y : 2D numpy array of target values. Dimensions (n x 1)
          weights : 2D numpy array of weights of the model. Dimensions ((d+1) x 1)

        Returns:
          dw : 2D numpy array of gradients w.r.t weights. Dimensions ((d+1) x 1)
        '''
        
        n = X.shape[0]     
        predictions = np.dot(X, weights)
        error = predictions - y
        dw = (1 / n) * np.dot(X.T, error)
        
        return dw

def plot_loss(error_list, total_epochs):
    '''
    This function plots the loss for each epoch.

    Args:
      error_list : list of validation loss for each epoch
      total_epochs : Total number of epochs
    Returns:
      None
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_epochs + 1), error_list, color='b')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE Loss')
    plt.title('Loss vs. Epochs')
    plt.grid(True)
    plt.show()

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
   

def generate_toy_dataset():
    '''
    This function generates a simple toy dataset containing 300 points with 1d feature 
    '''
    data = pd.read_csv('train.csv')
    
   
    X = data.iloc[:, 1:65].values  
    y = data.iloc[:, 65].values   
    
    
    X_train = X
    y_train = y
    y_train = y_train.reshape(-1, 1)
      # # Histogram 
    plt.figure(figsize=(10, 6))
    plt.hist(y_train, bins=30, alpha=0.7, color='green')
    plt.title('Target (y_train) Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
    
    #   ###############for test data
    data = pd.read_csv('test.csv')
    
    
    X_test_index=data.iloc[:, 0].values 
    X_test_index = X_test_index.reshape(-1, 1)
    X_test = data.iloc[:, 1:65].values  # Extract features (skipping the ID column)
    
    return X_train, y_train, X_test,X_test_index

def create_batches(X, y, batch_size):
    '''
    This function is used to create the batches of randomly selected data points.

    Args:
      X : 2D numpy array of data points. Dimensions (n x (d+1))
      y : 2D numpy array of target values. Dimensions (n x 1)

    Returns:
      batches : list of tuples with each tuple of size batch size.
    '''
    batches = []  
    
   
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Combine feature matrix X and target vector y into a single array
    data = np.hstack((X, y))
    np.random.shuffle(data)  
    num_batches = data.shape[0] // batch_size
    i=0
    for i in range(num_batches+1):
      if i<num_batches:  
        batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_batch = batch[:, :-1]
        Y_batch = batch[:, -1].reshape((-1, 1))
        batches.append((X_batch, Y_batch))
      if data.shape[0] % batch_size != 0 and i==num_batches:
        batch = data[i * batch_size:data.shape[0]]
        X_batch = batch[:, :-1]
        Y_batch = batch[:, -1].reshape((-1, 1))
        batches.append((X_batch, Y_batch))
    
    return batches

# Terminal text coloring
RESET = '\033[0m'
GREEN = '\033[32m'
RED = '\033[31m'
if __name__ == '__main__':
    
    X_train, y_train, X_test,X_test_index= generate_toy_dataset()


    # bias 
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

   
    model = LinearRegressionBatchGD(learning_rate=0.01, max_epochs=500, batch_size=500)
    model.fit(X_train, y_train, plot=True)

    # Predictions
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    y_hat_test_rounded = np.round(y_hat_test).astype(int)
      # Histogram for y_train
    plt.figure(figsize=(10, 6))
    plt.hist(y_hat_test_rounded, bins=30, alpha=0.7, color='green')
    plt.title('Target (y_test) Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
    data_j = np.hstack((X_test_index, y_hat_test_rounded))
    if data_j is None or data_j.size == 0:
        print("Data is not available for saving.")
    else:
        print(data_j.shape)
        np.savetxt('kaggle.csv', data_j, delimiter=',', header='ID,Score', comments='')

    plot_learned_equation(X_train, y_train, y_hat_train)
