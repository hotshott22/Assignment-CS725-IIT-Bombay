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

def solve():
    
  return max_indices
  



def solve():
    N,d = 9,5
    U,M1,M2 = initialise_input(N, d)
    
#     1 
    X = U @ M1.T
    Y = U @ M2.T
    
#     2
    X_hat = X + np.arange(1, N + 1).reshape(N, 1)
    
#     3
    Z = X @ X_hat.T  
    
    mask = np.ones((N, N), dtype=bool)
    mask[np.tril_indices(N, -1)] = False  
    Z_sparse = Z * mask
    
#     4
    exp_Z = np.exp(Z_sparse - np.max(Z_sparse, axis=1, keepdims=True))  
    Z_hat = exp_Z / exp_Z.sum(axis=1, keepdims=True)
    
#     5
    max_indices = np.argmax(Z_hat, axis=1)

    return max_indices

solve()


# max_indices = solve()
# print(max_indices)
