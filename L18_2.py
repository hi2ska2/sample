import math
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

N = 10000000

#A = np.zeros( (N+1, N+1) )
A = sparse.lil_matrix( (N+1, N+1) )
b = np.zeros( (N+1, 1  ) )

for ii in range(1,N):
    A[ii,ii+1] = 1.0
    A[ii,ii  ] = -2.0
    A[ii,ii-1] = 1.0

A[0,0] = 1.0
A[N,N] = 1.0
b[N] = 1.0

A = A.tocsr()
    
#update = np.linalg.solve(A, b)
update = sparse.linalg.spsolve(A,b)
update = update[:,None]

print(update)

