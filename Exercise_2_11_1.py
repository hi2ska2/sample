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
N = 120
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

A = np.zeros( (N+1, N+1) )
b = np.zeros( (N+1, 1  ) )

for inewton in range(1,10):
    print(inewton)
    for ii in range(1,N):    
        epsilon_l = epsilon_ox
        epsilon_r = epsilon_ox
        if ii>=interface1+1 and ii<=interface2:
            epsilon_l = epsilon_si
        if ii>=interface1 and ii<=interface2-1:
            epsilon_r = epsilon_si    

        b[ii] = epsilon_r*(phi[ii+1]-phi[ii])-epsilon_l*(phi[ii]-phi[ii-1])
        A[ii,ii-1] = epsilon_l
        A[ii,ii  ] = -epsilon_l-epsilon_r
        A[ii,ii+1] = epsilon_r

    for ii in range(interface1,interface2+1):
        control = 1.0
        if ii==interface1 or ii==interface2:
            control = 0.5
        hole[ii] = nint*math.exp(-phi[ii]/VT)
        elec[ii] = nint*math.exp( phi[ii]/VT)
        b[ii] = b[ii] + q*(hole[ii]-elec[ii]+Ndop)/epsilon0*l0*l0*control
        A[ii,ii] = A[ii,ii] - q*(hole[ii]+elec[ii])/VT/epsilon0*l0*l0*control

    b[0] = phi[0] - 0.33374
    A[0,0] = 1
    b[N] = phi[N] - 0.33374
    A[N,N] = 1

    update = np.linalg.solve(A, b)

    phi = phi - update

plt.plot(x/1e-9,elec/1e6,'bo-')
plt.xlabel('Position (nm)')
plt.ylabel('Electron density (/cm3)')
plt.show()
