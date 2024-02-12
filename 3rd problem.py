import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import newton, bisect, brentq
from scipy.constants import hbar, m_e
# Defining helper functions

# (1) Potential function

def V(r):
    V = (1/2)*k*r**2 + (1/3)*b*r**3
    return V

# (2) Defining A(r)

def A(r):
    a = (2*m_e/ħc**2)*(V(r) - E)
    return a

# (3) Defining Derivative functions of 'y'

def deriv(y, r):
    return np.array([y[1], A(r)*y[0]])

# (4) Defining wave function with respect to energy

def waveFunc(energy):
    global E
    global psi
    E = energy
    psi = odeint(deriv, psi0, r)
    return psi[-1, 0]

# (5) Finding zero of the wavefuncton by Shooting method

def zero(x, y):
    energyEigenValue = []
    s = np.sign(y)
    #s = np.array(s, dtype=int)
    for i in range(len(s)-1):
        if s[i] == -s[i+1]:
            zero = bisect(waveFunc, x[i], x[i+1])
            energyEigenValue.append(zero)
    return energyEigenValue

# Vectorizing the functions
V        = np.vectorize(V)
waveFunc = np.vectorize(waveFunc)
zero = np.vectorize(zero)
# Important parameters

m_e = 940
ħc = 197.3
k = 100
# Initialisation

N = 1000                              # Number of points on x-axis
psi = np.zeros([N, 2])                # To store wave function values and its derivative
psi0 = np.array([0.001, 0])           # wavefunction of initial staes

b = 30
r = np.linspace(-3, 3, N)              # x - axis
En = np.arange(0, 150, 1)
psiRight = waveFunc(En)
#psiRight
def zero(x, psi):
    energyEigenValue = []
    s = np.sign(psi)
    #s = np.array(s, dtype=int)
    for i in range(len(s)-1):
        if s[i] == -s[i+1]:
            zero = bisect(waveFunc, x[i], x[i+1])
            energyEigenValue.append(zero)
    return energyEigenValue
EigenValue = np.array(zero(En, psiRight))
EigenValue
array([ 31.55704024,  92.01776957, 144.81662013])
fig = plt.figure(figsize=(5, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
j=1
for i in range(len(EigenValue)):
    ax = fig.add_subplot(len(EigenValue), 1, j)
    waveFunc(EigenValue[i])
    plt.plot(r, psi.transpose()[0])
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    #plt.legend()
    plt.xlabel('$r$', fontsize=14)
    plt.ylabel('$\psi(r)$', fontsize=14)
    j+=1
plt.show()