import numpy as np

def initialise_input(N, d):
  '''
  N: Number of vectors
  d: dimension of vectors
  '''
  np.random.seed(0)
  U = np.random.randn(N, d)
  M1 = np.abs(np.random.randn(d, d))
  M2 = np.abs(np.random.randn(d, d))

  return U, M1, M2

def solve(N, d):
  
  U, M1, M2 = initialise_input(N, d)
  print("Matrix U:\n", U)
  print("Matrix M1:\n", M1)
  print("Matrix M2:\n", M2)

  '''
  Enter your code here for steps 1 to 6
  '''
  U, M1, M2 = initialise_input(N, d)
    
    # Step 1: Compute X
  X = U @ M1  # Matrix multiplication, resulting in shape (N, d)
    
    # Step 2: Compute Y
  Y = U @ M2  # Matrix multiplication, resulting in shape (N, d)
  print("Matrix X:\n", X)
   # Generate an array of integers from 1 to N
  array = np.arange(1, N + 1)  # Creates: [1, 2, 3]

  # Reshape this array into a column vector with shape (N, 1)
  offset = array.reshape(N, 1)  # Creates: [[1], [2], [3]]

  print("Matrix offset:\n", offset) 
    # Step 4: Modify X with the offset
  X_hat = X + offset  # Broadcasting to add offset to each row of X
  print("Matrix X_hat:\n", X_hat) 
    # If required, find indices of the maximum values
  max_indices = np.argmax(X, axis=1)  # Example, if needed for further analysis


  Y_T = Y.T  # Transpose of Y
  print("Matrix  Y:\n",  Y)
  print("MATRIX Y_TRANSPOSE:\n", Y_T) 
  Z = X @ Y_T  # Matrix multiplication: X * Y^T
  print("Matrix  Z:\n",  Z)
  '''
  N = Z.shape[0]
  sparsified_Z = np.zeros_like(Z)

  for i in range(N):
    for j in range(N):
      if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
        sparsified_Z[i, j] = Z[i, j]
  print("MATRIX sparsified_Z:\n",sparsified_Z) '''

  # Get the size of the matrix Z
  N = Z.shape[0]
  print("N:", N)  # Print the size of the matrix

  # Create row indices as a column vector
  row_indices = np.arange(N)[:, None]
  print("Row Indices:\n", row_indices)  # Print the column vector of row indices

  # Create column indices as a row vector
  col_indices = np.arange(N)
  print("Column Indices:\n", col_indices)  # Print the row vector of column indices

  # Create a mask where the condition (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0) is met
  mask = (row_indices % 2 == col_indices % 2).astype(int)
  print("Mask:\n", mask)  # Print the mask matrix

  # Apply mask to Z using broadcasting
  sparsified_Z = Z * mask
  print("Sparsified Z:\n", sparsified_Z)  # Print the sparsified matrix

  # Compute the exponential of each element in Z
  exp_Z = np.exp(Z)
  print("Exponential of Z:\n", exp_Z)  # Print the matrix of exponentials

  # Compute the sum of exponentials for each row
  row_sums = np.sum(exp_Z, axis=1, keepdims=True)  #axis=0: Operates along rows (sums columns). axis=1: Operates along columns (sums rows).
  print("Sum of exponentials (row-wise):\n", row_sums)  # Print the row sums

  # Compute the softmax values by dividing each exponential by the sum of exponentials for its row
  softmax_Z = exp_Z / row_sums
  print("Softmax Z:\n", softmax_Z)  # Print the resulting softmax matrix
  # Find the index of the maximum probability in each row
  max_indices = np.argmax(softmax_Z, axis=1)
  print("\nIndex of maximum probability in each row:")
  print(max_indices)
  return X,Y,max_indices

# Define N and d
'''N, d = 2, 2  # You can set N and d to any value you need'''
if __name__ == "__main__":
    # Take input from user
    N = int(input("Enter the number of vectors (N): "))
    d = int(input("Enter the dimension of vectors (d): "))

    # Call the solve function with user-provided N and d
    X, Y, max_indices = solve(N, d)

    # Print results
    #print("Matrix X:\n", X)
    #print("Matrix Y:\n", Y)
    #print("Max indices:\n", max_indices)


