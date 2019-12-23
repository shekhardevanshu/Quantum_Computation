import quantComp as qp
from math import cos, sin, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d


def initialize(N, alpha):
    # N = int(input("Number of qubits: "))
    # N = 2
    # numBasis = 2**N
    # psiOrig = list(map(float, input("Enter {0} coefficients of the state: ".format(numBasis)).rstrip().split()))
    psiOrig = [cos(alpha), 0, sin(alpha), 0]
    norm = sqrt(sum([i**2 for i in psiOrig]))
    psiOrig = list(map(lambda x: x / norm, psiOrig))
    return(np.asmatrix(psiOrig).transpose())


def delayed_choice():
    N = 2  # ancilla bit and system bit
    alpha = np.linspace(0, pi/2, 20)
    phi = np.linspace(-1/2, 3/2, 500)  # in the multiples of pi
    gate = qp.QGates(N)
    arr = []

    for alp in list(alpha):
        intensity = []
        for ph in list(phi):
            cnt = {}
            psi = initialize(N, alp)
            H = gate.Hadamard(2)
            psi = H @ psi
            phase = gate.phase(ph, 2, False)
            psi = phase @ psi
            ContrH = gate.Cgate(gate.H)
            psi = ContrH @ psi
            numMeas = 500
            states = qp.QuantuMeasure(N, psi, numMeas)
            for st in states:
                cnt[st[2]] = cnt.get(st[2], 0) + 1
            try:
                intensity.append(cnt['0'] / numMeas)
            except:
                intensity.append(0)
        arr.append(intensity)
        # print(f'{alp = }')
        # print(intensity)

    # plotting
    arr = np.array(arr)
    plt.ion()  # turning on interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(phi, arr[0, :])
    plt.ylim((0, 1))
    plt.xlabel(r'$\phi $')
    plt.ylabel(r'$I_0$')
    plt.title(r'Particle nature to wave nature by changing $\alpha $')

    for _ in range(2):
        for k in range(len(alpha)):
            line1.set_ydata(arr[k, :])
            fig.canvas.draw()
            plt.pause(0.3)


if __name__ == '__main__':
    delayed_choice()
