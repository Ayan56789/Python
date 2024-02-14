import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import newton, bisect, brentq
from scipy.constants import hbar, m_e
# Defining helper functions

# (1) Potential function

def V(r, r0, α, D):
    V = D*(1 - np.exp(-α*(r - r0)))**2
    return V

# (2) Defining A(r)

def A(r):
    a = (2*μ/ħc**2)*(V(r, r0, α, D) - E)
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

μ = 940e6
D = 0.755501
α = 1.44
r0 = 0.131349
ħc = 1973

# Initialisation

N = 100                              # Number of points on x-axis
psi = np.zeros([N, 2])                # To store wave function values and its derivative
psi0 = np.array([0.001, 2])           # wavefunction of initial staes


r = np.linspace(-0.2, 0.2, 100)              # x - axis
En = np.arange(0, 15, 1)
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
print(EigenValue)

fig = plt.figure(figsize=(5, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
j=1
for i in range(len(EigenValue)):
    ax = fig.add_subplot(len(EigenValue), 1, j)
    waveFunc(EigenValue[i])
    plt.plot(r, psi.transpose()[0])
    plt.axhline(lw=2.3, c='black')
    plt.axvline(lw=2.3, c='black')
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.legend(fontsize=15)
    plt.xlabel('$r$', fontsize=14)
    plt.ylabel('$\psi(r)$', fontsize=14)
    j+=1
plt.show()
