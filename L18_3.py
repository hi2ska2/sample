import math
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

def Ber(x):
    c = abs(x)
    if (c < 2.502e-2):
        x2 = x*x
        return 1.0-x/2.0+x2/12.0*(1.0-x2/60.0*(1.0-x2/42.0))
    elif (c < 1.5e-1):
        x2 = x*x
        return 1.0-x/2.0+x2/12.0*(1.0-x2/60.0*(1.0-x2/42.0*(1.0-x2/40.0*(1.0-0.02525252525252525252525*x2))))       
    elif (x > 150.01):
        return x*math.exp(-x)
    else:
        return x/(math.exp(x)-1.0)

def dBer(x):
    c = abs(x)
    if (c < 2.502e-2):
        x2 = x*x
        return -0.5+x/6.0*(1.0-x2/30.0*(1.0-x2/28.0))
    elif (c < 1.5e-1):
        x2 = x*x
        return -0.5+x/6.0*(1.0-x2/30.0*(1.0-x2/28.0*(1.0-x2/30.0*(1.0-0.03156565656565656565657*x2))))
    elif (x > 150.01):
        return math.exp(-x) - Ber(x)
    else:
        inv_expx_1 = 1.0/(math.exp(x)-1.0)
        return inv_expx_1-Ber(x)*(inv_expx_1+1.0)

q = 1.602192e-19
epsilon0 = 8.854187817e-12
nint = 1.075e16
kB = 1.38065e-23 # Boltzmann constant
T = 300.0
VT = kB*T/q
Dp = 470.5 * 1e-4 * VT 
Dn = 1417 * 1e-4 * VT 

# L is the length in m.
L = 400e-9
N = 10000
ijunction = N // 2
l0 = L/N
epsilon_si = 11.7
ND = 1e17
ND = ND * 1e6
phiD = VT*math.asinh(0.5*ND/nint)
NA = 1e17
NA = NA * 1e6
phiA = -VT*math.asinh(0.5*NA/nint)

x = np.arange(N+1)* L / N
dop = np.zeros ( (N+1,1) )
dop[0:ijunction] = ND
dop[ijunction+1:N+1] = -NA

phi = np.zeros( (N+1, 1  ) )
for ii in range(0,N+1):
    phi[ii] = VT*math.asinh(0.5*dop[ii]/nint)
hole = np.zeros( (N+1, 1  ) )
elec = np.zeros( (N+1, 1  ) )

#####################
# Nonlinear Poisson #
##################### 

for inewton in range(1,10):

    #A = np.zeros( (3*(N+1), 3*(N+1)) )
    A = sparse.lil_matrix( (3*(N+1), 3*(N+1)) )
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

    A = A.tocsr()

    #update = np.linalg.solve(A, b)
    update = sparse.linalg.spsolve(A, b)
    update = update[:,None]

    print( inewton, np.linalg.norm(update[range(0,3*(N+1),3)],np.inf) )
    
    phi  = phi  - update[range(0,3*(N+1),3)]
    hole = hole - update[range(1,3*(N+1),3)]
    elec = elec - update[range(2,3*(N+1),3)]

#plt.plot(x/1e-9,dop/1e6,'bo-')
#plt.xlabel('Position (nm)')
#plt.ylabel('Doping concentration (/cm3)')
#plt.show()

#plt.plot(x/1e-9,phi,'bo-')
#plt.xlabel('Position (nm)')
#plt.ylabel('Electrostatic potential (V)')
#plt.ylim(-0.7,0.7)
#plt.show()

###################
# Drift-diffusion #
################### 

IV = np.zeros( (21,4) )

for ibias in range(0,21):

    Vapplied = 0.05*ibias
    print(Vapplied)
    IV[ibias,0] = Vapplied

    for inewton in range(1,40):

        #A = np.zeros( (3*(N+1), 3*(N+1)) )
        A = sparse.lil_matrix( (3*(N+1), 3*(N+1)) )
        b = np.zeros( (3*(N+1), 1      ) )  
    
        for ii in range(1,N):    

            # Poisson
        
            b[3*ii] = epsilon_si*(phi[ii+1]-phi[ii])-epsilon_si*(phi[ii]-phi[ii-1])
            A[3*ii,3*(ii-1)] = epsilon_si
            A[3*ii,3* ii   ] = -epsilon_si-epsilon_si
            A[3*ii,3*(ii+1)] = epsilon_si

            b[3*ii] = b[3*ii] + q*(hole[ii]-elec[ii]+dop[ii])/epsilon0*l0*l0
            A[3*ii,3*ii+1] =  q/epsilon0*l0*l0
            A[3*ii,3*ii+2] = -q/epsilon0*l0*l0

            # Hole continuity

            dphi = (phi[ii+1]-phi[ii])/VT
            b[3*ii+1] = -q*Dp/l0*( hole[ii+1]*Ber(-dphi) - hole[ii]*Ber(dphi) )
            A[3*ii+1,3*(ii+1)+1] = A[3*ii+1,3*(ii+1)+1] - q*Dp/l0*(  Ber(-dphi) )
            A[3*ii+1,3* ii   +1] = A[3*ii+1,3* ii   +1] - q*Dp/l0*( -Ber( dphi) )
            A[3*ii+1,3*(ii+1)  ] = A[3*ii+1,3*(ii+1)  ] - q*Dp/l0*( -hole[ii+1]*dBer(-dphi) - hole[ii]*dBer(dphi) ) / VT
            A[3*ii+1,3* ii     ] = A[3*ii+1,3* ii     ] - q*Dp/l0*(  hole[ii+1]*dBer(-dphi) + hole[ii]*dBer(dphi) ) / VT        

            dphi = (phi[ii]-phi[ii-1])/VT
            b[3*ii+1] = b[3*ii+1] + q*Dp/l0*( hole[ii]*Ber(-dphi) - hole[ii-1]*Ber(dphi) )
            A[3*ii+1,3* ii   +1] = A[3*ii+1,3* ii   +1] + q*Dp/l0*(  Ber(-dphi) )
            A[3*ii+1,3*(ii-1)+1] = A[3*ii+1,3*(ii-1)+1] + q*Dp/l0*( -Ber( dphi) )
            A[3*ii+1,3* ii     ] = A[3*ii+1,3* ii     ] + q*Dp/l0*( -hole[ii]*dBer(-dphi) - hole[ii-1]*dBer(dphi) ) / VT
            A[3*ii+1,3*(ii-1)  ] = A[3*ii+1,3*(ii-1)  ] + q*Dp/l0*(  hole[ii]*dBer(-dphi) + hole[ii-1]*dBer(dphi) ) / VT

            # Electron continuity

            dphi = (phi[ii+1]-phi[ii])/VT
            b[3*ii+2] = q*Dn/l0*( elec[ii+1]*Ber(dphi) - elec[ii]*Ber(-dphi) )
            A[3*ii+2,3*(ii+1)+2] = A[3*ii+2,3*(ii+1)+2] + q*Dn/l0*(  Ber( dphi) )
            A[3*ii+2,3* ii   +2] = A[3*ii+2,3* ii   +2] + q*Dn/l0*( -Ber(-dphi) )
            A[3*ii+2,3*(ii+1)  ] = A[3*ii+2,3*(ii+1)  ] + q*Dn/l0*(  elec[ii+1]*dBer(dphi) + elec[ii]*dBer(-dphi) ) / VT
            A[3*ii+2,3* ii     ] = A[3*ii+2,3* ii     ] + q*Dn/l0*( -elec[ii+1]*dBer(dphi) - elec[ii]*dBer(-dphi) ) / VT       

            dphi = (phi[ii]-phi[ii-1])/VT
            b[3*ii+2] = b[3*ii+2] - q*Dn/l0*( elec[ii]*Ber(dphi) - elec[ii-1]*Ber(-dphi) )
            A[3*ii+2,3* ii   +2] = A[3*ii+2,3* ii   +2] - q*Dn/l0*(  Ber( dphi) )
            A[3*ii+2,3*(ii-1)+2] = A[3*ii+2,3*(ii-1)+2] - q*Dn/l0*( -Ber(-dphi) )
            A[3*ii+2,3* ii     ] = A[3*ii+2,3* ii     ] - q*Dn/l0*(  elec[ii]*dBer(dphi) + elec[ii-1]*dBer(-dphi) ) / VT
            A[3*ii+2,3*(ii-1)  ] = A[3*ii+2,3*(ii-1)  ] - q*Dn/l0*( -elec[ii]*dBer(dphi) - elec[ii-1]*dBer(-dphi) ) / VT       
        
        b[0] = phi[0] - phiD
        A[0,0] = 1.0
        b[3*N] = phi[N] - phiA - Vapplied
        A[3*N,3*N] = 1.0

        b[1] = hole[0] - nint*math.exp(-phiD/VT)
        A[1,1] = 1.0
        b[3*N+1] = hole[N] - nint*math.exp(-phiA/VT)
        A[3*N+1,3*N+1] = 1.0

        b[2] = elec[0] - nint*math.exp(phiD/VT)
        A[2,2] = 1.0
        b[3*N+2] = elec[N] - nint*math.exp(phiA/VT)
        A[3*N+2,3*N+2] = 1.0

        A = A.tocsr()

        #update = np.linalg.solve(A, b)
        update = sparse.linalg.spsolve(A,b,False)
        update = update[:,None]

        phi  = phi  - update[range(0,3*(N+1),3)]
        hole = hole - update[range(1,3*(N+1),3)]
        elec = elec - update[range(2,3*(N+1),3)]

        phiNorm = np.linalg.norm(update[range(0,3*(N+1),3)],np.inf)

        print( inewton, np.linalg.norm(update[range(0,3*(N+1),3)],np.inf), np.linalg.norm(update[range(1,3*(N+1),3)],np.inf), np.linalg.norm(update[range(2,3*(N+1),3)],np.inf) )

        if phiNorm<1e-10:
            #jj = ijunction
            jj = N
            dphi = (phi[jj]-phi[jj-1])/VT
            IV[ibias,1] =  q*Dp/l0*( hole[jj]*Ber(-dphi) - hole[jj-1]*Ber( dphi) )
            IV[ibias,2] = -q*Dn/l0*( elec[jj]*Ber( dphi) - elec[jj-1]*Ber(-dphi) )
            IV[ibias,3] = IV[ibias,1] + IV[ibias,2]
            break

#print(phi)
#print(hole)
#print(elec)

#plt.plot(x/1e-9,elec/1e6,'bo-')
#plt.plot(x/1e-9,hole/1e6,'rs-')
#plt.xlabel('Position (nm)')
#plt.ylabel('Carrier density (/cm3)')
#plt.yscale('log')
#plt.show()
        
print(IV)

#jp = np.zeros( (N,1) )
#jn = np.zeros( (N,1) )
#for ii in range(1,N+1):
#    dphi = (phi[ii]-phi[ii-1])/VT
#    jp[ii-1] =  q*Dp/l0*( hole[ii]*Ber(-dphi) - hole[ii-1]*Ber( dphi) )
#    jn[ii-1] = -q*Dn/l0*( elec[ii]*Ber( dphi) - elec[ii-1]*Ber(-dphi) )

#plt.plot(abs(jp[::-1])*1e-4,'r')
#plt.plot(abs(jn[::-1])*1e-4,'b')
#plt.show()

plt.plot(IV[:,0],IV[:,1]/1e12,'bo-')
plt.xlabel('Voltage (V)')
#plt.ylabel('Newton iteration (1)')
plt.ylabel('Anode current (A/um2)')
#plt.ylim(0,6)
plt.yscale('log')
plt.show()
        
