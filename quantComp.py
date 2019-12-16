# Project -> Simulate a quantum computer
# Author: Devanshu Shekhar
# Date: 2/10/19

import random
from math import sqrt, log, log2
from cmath import exp, pi
import numpy as np
# import matplotlib.pyplot as plt

class QGates():
    def __init__(self, N):
        self.n = N  #total number of qubits    
        self.I = np.identity(2) #identity matrix
        self.H =  np.matrix("1,1;1,-1")/sqrt(2) #Hadamard gate
        self.NOT =  np.matrix("0,1;1,0") #NOT gate
        self.CNOT = np.matrix("1 0 0 0;0 1 0 0;0 0 0 1;0 0 1 0")
        # self.Zeros = np.zeros((2,2))

    def delta(self,i,j):
        if i == j: return(1)
        else: return(0)

    '''def DtensorMult(self,M,a):
        genM = []
        n = self.n
        for i in range(2**n):
            l = []
            for j in range(2**n):
                x,y = int2bin(n, i), int2bin(n, j)
                m = 1
                for q in range(n):
                    if q == a-1: m *= M[int(x[q]),int(y[q])]
                    else: m *= self.delta(int(x[q]),int(y[q]))
                    # print(q, j, int(x[q]),int(y[q]))
                print(i, j, m)
                l.append(m)
            genM.append(l)
        return(np.matrix(genM))'''

    def tensorMult(self,M,a):
        n = self.n
        N = 2 ** n
        genM = np.asmatrix(np.zeros((N, N), dtype=np.float16))
        # genM = [[0.0] * N] * N
        # print(genM)
        for i in range(N):
            # l = []
            for j in [i, min(i + (2 ** (n - a)), N - 1)]:
                # x,y = int2bin(n, i), int2bin(n, j)
                x, y = i, j
                b_x = (x >> (n - a)) & 1
                b_y = (y >> (n - a)) & 1
                m = M[b_x, b_y]

                # for c in range(1, n + 1):
                #     d = z & 1
                #     b_x = x & 1
                #     b_y = y & 1
                #     if c == n - a + 1:
                #         m *= M[b_x, b_y]
                #     else:
                #         m *= int(not d)
                #         if d == 1:
                #             break
                #     z >>= 1
                #     x >>= 1
                #     y >>= 1

                '''for q in range(n):
                    if q == a-1: m *= M[int(x[q]),int(y[q])]
                    else: m *= self.delta(int(x[q]),int(y[q]))'''
                    # print(q, j, int(x[q]),int(y[q]))
                # print(m)
                genM[i,j] = m
                genM[j,i] = m
                # print(i, j, m)
                # print(genM)
        # return np.matrix(genM)
        return genM

    def Hadamard(self, i):
        return(self.tensorMult(self.H,i))

    def phase(self,theta, i=1, Rth = True):    #theta is taken as the multiple of pi
        # Remember: non-default argument(theta) mustn't follow default arguments(i,Rth)
        eitheta = exp(1j*(theta*pi))
        if round(eitheta.imag, 5) == 0: e = eitheta.real
        else: e = complex(eitheta.real, round(eitheta.imag, 5))
        
        Rtheta = np.matrix([[1, 0], [0, e]])
        if Rth == True:
            return(Rtheta)
        else:
            return(self.tensorMult(Rtheta,i))
    def Cgate(self, M):
        return( np.matrix([ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, M[0,0], M[0,1]], [0, 0, M[1,0],M[1,1]] ]) )


def int2bin(N,n):
    l = [0]*N
    i = N
    while(True):
        l[i-1] = n%2
        n = n//2
        i -= 1
        if n == 0: break
    l = list(map(str,l))
    return(''.join(l))

def initialize():
    N = int(input("Number of qubits: "))
    # N = 2
    
    numBasis = 2**N
    psiOrig = list(map(float, input("Enter {0} coefficients of the state: ".format(numBasis)).rstrip().split()))
    if not any(psiOrig):
        psiOrig = [0]*numBasis
        psiOrig[0] = 1
    norm = sqrt(sum([i**2 for i in psiOrig]))
    psiOrig = list(map(lambda x: x / norm, psiOrig))
    return(psiOrig, N)

def measurement(psiOrig, N):

    # N = int(log(len(psiOrig))/log(2))
    psi = list(psiOrig)

    while True:
        r = round(random.random(),5)
        q = psi[0]**2
        state = None
        for j in range(0,len(psi)):           
            if j == len(psi)-1:  
                # collapsedState.append('|'+int2bin(N,j)+'>')
                print('|'+int2bin(N,j)+'>\n')
                state = j
                break
            elif r <= q:               
                # collapsedState.append('|'+int2bin(N,j)+'>')
                print('|'+int2bin(N,j)+">\n")
                state = j
                break
            q += psi[j+1]**2

        inp = int(input("Enter \n 1. Measure same state \n 2. Restart measurement \n 3. exit \n"))
        if inp == 1:
            psi = [0]*len(psi)
            psi[state] = 1
        elif inp == 2:
            psi = list(psiOrig)
        else: break

def genCGate(N,M,a,b):
    gate = QGates(N)
    Cgate = gate.Cgate(M)
    genCgate = []
    for i in range(2**N):
        l = []
        for j in range(2**N):
            x, y = int2bin(N,i), int2bin(N,j)
            # print(a-1, b-1)
            l.append(Cgate[int(x[a-1]+x[b-1], 2), int(y[a-1]+y[b-1], 2)])
            for p in range(len(x)):
                if p != a-1 and p != b-1:       
                    l[j] *= gate.delta(x[p],y[p])
        genCgate.append(l)
    return(np.matrix(genCgate))

def fullQuntumComp():
    #intialization
    psi, N = initialize()
    psi = np.matrix(psi).transpose()

    #gates
    gate = QGates(N)
    while True:
        g = int(input("Select gate:\n 1. Hadamard gate\n2. Phase shift gate\n3.Done\n"))
        if g == 3: break
        i = int(input("apply on qubit: "))
        if g == 1:
            psi = np.matmul(gate.Hadamard(i), psi)
            # print(gate.Hadamard(i))
            # print(psi)
        elif g == 2:
            theta = int(input("Enter theta in multiples of pi: "))
            psi = np.matmul(gate.phase(i,theta), psi)

    measurement(psi, N)

def QuantuMeasure(psi, numMeas):
    # psi, N = initialize()
    # numMeas = int(input("How many measurements: "))
    N = int(log2(len(psi)))
    collapsedState = []
    # cnt = {}

    for _ in range(numMeas):
        r = random.random()
        q = psi[0]**2
        
        for j in range(0,len(psi)):           
            if j == len(psi)-1:  
                collapsedState.append('|'+int2bin(N,j)+'>')
                # print('|'+int2bin(N,j)+'>\n')
                break
            elif r <= q:               
                collapsedState.append('|'+int2bin(N,j)+'>')
                # print('|'+int2bin(N,j)+">\n")
                break
            q += psi[j+1]**2
    return(collapsedState)
    # for st in collapsedState:
        # cnt[st] = cnt.get(st, 0) + 1
    # print(cnt)
    # plt.hist(cnt)
    # plt.show()





if __name__ == '__main__':
    # print(int2bin(3,6))
    # psi, N = initialize()
    # psi = np.matrix(psi).transpose()
    # print(psi)
    # print(np.shape(psi))
    # measurement(psi)
    gate = QGates(3)
    h = gate.Hadamard(2)
    print(h)
    # for i in h:
    #     for j in i:
    #         print(j , end=" ")
    #     print()
    # print(gate.CNOT[0,0])
    # N = gate.phase(theta = 1)
    # print(N)
    # print(genCGate(2,N,2,1))
    # print(np.shape(H))
    # P = gate.phase(3,1)
    # print(P)
    # fullQuntumComp()
    # QuantuMeasure()
    # print(genCNOT(3,2,1))