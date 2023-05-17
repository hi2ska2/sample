import math
import numpy as np
import matplotlib.pyplot as plt

def bernoulli(x):
    return x/(math.exp(x)-1.0)

def dBdx(x):
    inv_expx_1 = 1.0/(math.exp(x)-1.0)
    return inv_expx_1-bernoulli(x)*(inv_expx_1+1.0)

N = 21
data = np.zeros( (N+1,2))
for ii in range(0,N+1):
    x = -4.0 + 8.0/N * ii
    print(x)
    #data[ii,:] = [x, bernoulli(x)]
    data[ii,:] = [x, dBdx(x)]

plt.plot(data[:,0],data[:,1],'bo-')
plt.xlabel('x (1)')
plt.ylabel('dB/dx(x) (1)')
plt.ylim([-1,0])
plt.show()
