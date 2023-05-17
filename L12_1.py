import math
import numpy as np
import matplotlib.pyplot as plt

q = 1.602192e-19
epsilon0 = 8.854187817e-12
nint = 1e16
kB = 1.38065e-23 # Boltzmann constant
T = 300.0
VT = kB*T/q

# L is the length in m.
L = 400e-9
N = 400
ijunction = 200
l0 = L/N
epsilon_si = 11.7
ND = 1e20
ND = ND * 1e6
phiD = VT*math.asinh(0.5*ND/nint)
NA = 1e19
NA = NA * 1e6
phiA = -VT*math.asinh(0.5*NA/nint)

x = np.arange(N+1)* L / N
dop = np.zeros ( (N+1,1) )
dop[0:ijunction] = ND
dop[ijunction:N+1] = -NA

phi = np.zeros( (N+1, 1  ) )
for ii in range(0,N+1):
    phi[ii] = VT*math.asinh(0.5*dop[ii]/nint)
hole = np.zeros( (N+1, 1  ) )
elec = np.zeros( (N+1, 1  ) )

for inewton in range(1,10):

    A = np.zeros( (3*(N+1), 3*(N+1)) )
    b = np.zeros( (3*(N+1), 1      ) )  
    
    for ii in range(1,N):    
        
        b[3*ii] = epsilon_si*(phi[ii+1]-phi[ii])-epsilon_si*(phi[ii]-phi[ii-1])
        A[3*ii,3*(ii-1)] = epsilon_si
        A[3*ii,3* ii   ] = -epsilon_si-epsilon_si
        A[3*ii,3*(ii+1)] = epsilon_si

        b[3*ii+1] = hole[ii]
        A[3*ii+1,3*ii+1] = 1.0

        b[3*ii+2] = elec[ii]
        A[3*ii+2,3*ii+2] = 1.0
    
        b[3*ii] = b[3*ii] + q*(hole[ii]-elec[ii]+dop[ii])/epsilon0*l0*l0
        A[3*ii,3*ii+1] =  q/epsilon0*l0*l0
        A[3*ii,3*ii+2] = -q/epsilon0*l0*l0
        b[3*ii+1] = b[3*ii+1] - nint*math.exp(-phi[ii]/VT)
        A[3*ii+1,3*ii] =  nint/VT*math.exp(-phi[ii]/VT)
        b[3*ii+2] = b[3*ii+2] - nint*math.exp( phi[ii]/VT)
        A[3*ii+2,3*ii] = -nint/VT*math.exp( phi[ii]/VT)

    b[0] = phi[0] - phiD
    A[0,0] = 1.0
    b[3*N] = phi[N] - phiA
    A[3*N,3*N] = 1.0

    b[1] = hole[0] - nint*math.exp(-phiD/VT)
    A[1,1] = 1.0
    b[3*N+1] = hole[N] - nint*math.exp(-phiA/VT)
    A[3*N+1,3*N+1] = 1.0

    b[2] = elec[0] - nint*math.exp(phiD/VT)
    A[2,2] = 1.0
    b[3*N+2] = elec[N] - nint*math.exp(phiA/VT)
    A[3*N+2,3*N+2] = 1.0

    update = np.linalg.solve(A, b)

    print( inewton, np.linalg.norm(update[range(0,3*(N+1),3)],np.inf) )
    
    phi  = phi  - update[range(0,3*(N+1),3)]
    hole = hole - update[range(1,3*(N+1),3)]
    elec = elec - update[range(2,3*(N+1),3)]

#plt.plot(x/1e-9,dop/1e6,'bo-')
#plt.xlabel('Position (nm)')
#plt.ylabel('Doping concentration (/cm3)')
#plt.show()

plt.plot(x/1e-9,phi,'bo-')
plt.xlabel('Position (nm)')
plt.ylabel('Electrostatic potential (V)')
plt.ylim(-0.7,0.7)
plt.show()
