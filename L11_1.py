import math
import numpy as np
import matplotlib.pyplot as plt

q = 1.602192e-19
epsilon0 = 8.854187817e-12
nint = 1e16
kB = 1.38065e-23 # Boltzmann constant
T = 300.0
VT = kB*T/q

# a is the length in m.
a = 6e-9
tox = 0.5e-9
N = 12
l0 = a/N
iox = round(tox/l0)
interface1 = iox
interface2 = N-iox
epsilon_si = 11.7
epsilon_ox = 3.9
Ndop = -1e24

x = np.arange(N+1)* a / N

phi = np.zeros( (N+1, 1  ) )
hole = np.zeros( (N+1, 1  ) )
elec = np.zeros( (N+1, 1  ) )

for inewton in range(1,10):

    A = np.zeros( (3*(N+1), 3*(N+1)) )
    b = np.zeros( (3*(N+1), 1      ) )  
    
    for ii in range(1,N):    
        epsilon_l = epsilon_ox
        epsilon_r = epsilon_ox
        if ii>=interface1+1 and ii<=interface2:
            epsilon_l = epsilon_si
        if ii>=interface1 and ii<=interface2-1:
            epsilon_r = epsilon_si    

        b[3*ii] = epsilon_r*(phi[ii+1]-phi[ii])-epsilon_l*(phi[ii]-phi[ii-1])
        A[3*ii,3*(ii-1)] = epsilon_l
        A[3*ii,3* ii   ] = -epsilon_l-epsilon_r
        A[3*ii,3*(ii+1)] = epsilon_r

        b[3*ii+1] = hole[ii]
        A[3*ii+1,3*ii+1] = 1.0

        b[3*ii+2] = elec[ii]
        A[3*ii+2,3*ii+2] = 1.0

    for ii in range(interface1,interface2+1):
        control = 1.0
        if ii==interface1 or ii==interface2:
            control = 0.5
        b[3*ii] = b[3*ii] + q*(hole[ii]-elec[ii]+Ndop)/epsilon0*l0*l0*control
        A[3*ii,3*ii+1] =  q/epsilon0*l0*l0*control
        A[3*ii,3*ii+2] = -q/epsilon0*l0*l0*control
        b[3*ii+1] = b[3*ii+1] - nint*math.exp(-phi[ii]/VT)
        A[3*ii+1,3*ii] =  nint/VT*math.exp(-phi[ii]/VT)
        b[3*ii+2] = b[3*ii+2] - nint*math.exp( phi[ii]/VT)
        A[3*ii+2,3*ii] = -nint/VT*math.exp( phi[ii]/VT)

    b[0] = phi[0] - 0.33374
    A[0,0] = 1.0
    b[3*N] = phi[N] - 0.33374
    A[3*N,3*N] = 1.0

    b[1] = hole[0]
    A[1,1] = 1.0
    b[3*N+1] = hole[N]
    A[3*N+1,3*N+1] = 1.0

    b[2] = elec[0]
    A[2,2] = 1.0
    b[3*N+2] = elec[N]
    A[3*N+2,3*N+2] = 1.0

    update = np.linalg.solve(A, b)

    print( inewton, np.linalg.norm(update[range(0,3*(N+1),3)],np.inf) )
    
    phi  = phi  - update[range(0,3*(N+1),3)]
    hole = hole - update[range(1,3*(N+1),3)]
    elec = elec - update[range(2,3*(N+1),3)]

plt.plot(x/1e-9,elec/1e6,'bo-')
plt.xlabel('Position (nm)')
plt.ylabel('Electron density (/cm3)')
plt.show()
