import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import bisect
# Basic parameters
E0 = -12       # approximate ground state energy
e = 3.795      # charge of electron in eVA
hbarc = 1973   # in eVA
m = 0.511e6    # in eV/c^2
a = 5          # in units of angstrum

# Based on the given parameters, calculate 2m/hbar^2c
C = 2*m/hbarc**2
# Defining Helper Functions

def V(r):
    return - (e**2/r )*np.exp(-r/a)

def A(r):
    return C*(V(r)-E)

def dzdr(z, r):
    x, y = z
    dxdr = y
    dydr = A(r)*x
    dzdr = np.array([dxdr, dydr])
    return dzdr

def waveFunc(energy):
    global sol
    global E
    E = energy
    sol = odeint(dzdr, z[0], r)
    return sol[-1, 0]
# setting arrays for storing values

r = np.arange(1e-15, 10, 0.01)
z = np.zeros([len(r),2])
x0 = 0.001
y0 = 1
z[0] = [x0, y0]
# Finding energy eigen values

Energy = np.linspace(-20, 0, 100)
waveFuncRight = np.array([waveFunc(Eval) for Eval in Energy])
sign = np.sign(waveFuncRight)
eigenValue = []
for i in range(len(sign)-1):
    if sign[i] == - sign[i+1]:
        en = bisect(waveFunc, Energy[i], Energy[i+1])
        eigenValue += [en]

print('Energy Eigen Values are \n'+str(eigenValue))
Energy Eigen Values are
[-11.06410939502324, -1.284295766697573]
# Plotting wavefunctions

def plot():
    fig, ax = plt.subplots(figsize=(8, 5))
    # set the x-spine
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

plot()

color = ['red', 'blue', 'orange']
for i in range(len(eigenValue)):
    waveFunc(eigenValue[i])
    plt.plot(r, sol[:,0], color=color[i], label='n = '+str(i)+'   E'+str(i)+' = '+str(round(eigenValue[i],1)))

plt.xlabel('r')
plt.ylabel('$\psi(r)$')
plt.text(8,0.08, '$V(r)=\\frac{e^2}{r}e^{-r/a};  \\ a$ = '+str(a))
plt.legend()
#plt.grid()
plt.show()