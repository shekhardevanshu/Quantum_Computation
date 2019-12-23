# Project -> Simulate a quantum computer
# Author: Devanshu Shekhar
# Date: 2/10/19

import random
from math import sqrt, log, log2
from cmath import exp, pi
import numpy as np
from scipy.sparse import coo_matrix

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

    def tensorMult(self, M, a, dtype):
        n = self.n
        N = 2 ** n
        col = []; row = []; data = []
        for i in range(N):
            # l = []
            ls = [i]
            if i + (2 ** (n - a)) < N:  ls.append(i + (2 ** (n - a)))
            for j in ls:
                # x,y = int2bin(n, i), int2bin(n, j)
                x, y, z = i, j, i ^ j
                b_x = (x >> (n - a)) & 1
                b_y = (y >> (n - a)) & 1
                m = M[b_x, b_y]

                for c in range(1, n + 1):
                    if c != a:
                    # print(d, m)
                        d = (z >> n-c) & 1
                        m *= int(not d)
                        if d == 1: break
                if m != 0:
                    row.append(i)
                    col.append(j)
                    data.append(m)
                    if i != j:
                        row.append(j)
                        col.append(i)
                        data.append(m)
        genM = coo_matrix((data, (row, col)), shape=(N, N)) #sparse matrix implementation
        return genM

    def Hadamard(self, i):
        return(self.tensorMult(self.H, i, dtype=float))

    def phase(self,theta, i=1, Rth = True):    #theta is taken as the multiple of pi
        # Remember: non-default argument(theta) mustn't follow default arguments(i,Rth)
        eitheta = exp(1j*(theta*pi))
        # if round(eitheta.imag, 5) == 0: e = eitheta.real
        # elif round(eitheta.real, 5) == 0: e = eitheta.imag
        # else: e = complex(round(eitheta.real, 5), round(eitheta.imag, 5))
        e = complex(round(eitheta.real, 5), round(eitheta.imag, 5))
        
        Rtheta = np.matrix([[1, 0], [0, e]])
        if Rth == True:
            return(Rtheta)
        else:
            return(self.tensorMult(Rtheta,i, dtype=complex))
    def Cgate(self, M):
        return( np.matrix([ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, M[0,0], M[0,1]], [0, 0, M[1,0],M[1,1]] ]) )


def int2bin(N,n):
    # print(N)
    l = [0]*N
    i = N
    
    while(True):
        # print(i,N)
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
    return(np.asmatrix(psiOrig).transpose(), N)

def measurement(psiOrig, N):

    # N = int(log(len(psiOrig))/log(2))
    psi = list(psiOrig)

    while True:
        r = round(random.random(),5)
        q = abs(psi[0])**2
        for j in range(0,len(psi)):           
            if j == len(psi)-1:  
                # collapsedState.append('|'+f'{j:0{N}b}'+'>')
                print('|'+f'{j:0{N}b}'+'>\n')
                state = j
                break
            elif r <= q:               
                # collapsedState.append('|'+f'{j:0{N}b}'+'>')
                print('|'+f'{j:0{N}b}'+">\n")
                state = j
                break
            q += abs(psi[j+1])**2

        inp = int(input("Enter \n 1. Measure same state \n 2. Restart measurement \n 3. exit \n"))
        if inp == 1:
            psi = [0]*len(psi)
            psi[state] = 1
        elif inp == 2:
            psi = list(psiOrig)
        else: break

def genCGate(N,M,a,b):
    NN = 2**N
    gate = QGates(N)
    Cgate = gate.Cgate(M)
    # print(Cgate)
    genCgate = []
    row, col, data = [], [], []
    for i in range(NN):
        x = i
        b_x1, b_x2 = (x >> N-a) & 1, (x >> N-b) & 1
        for j in range(i, NN):
            # x, y = int2bin(N,i), int2bin(N,j)
            y, z = j, i ^ j
            # print(z)
            # print(a-1, b-1)
            # l.append(Cgate[int(x[a-1]+x[b-1], 2), int(y[a-1]+y[b-1], 2)])    
            b_y1, b_y2 = (y >> N-a) & 1, (y >> N-b) & 1
            m = Cgate[int(str(b_x1)+str(b_x2), 2), int(str(b_y1)+str(b_y2), 2)]
            # print(m, i, j)
            for p in range(1, N+1):
                
                if p != a and p != b:
                    # print(d, m)
                    # print('delta')
                    d = (z >> N-p) & 1
                    m *= int(not d)
                    if d == 1: break
                # z >>= 1
            if m != 0:
                row.append(i)
                col.append(j)
                data.append(m)
            #by symmetric property
                if i != j:
                    row.append(j)
                    col.append(i)
                    data.append(m)
                # print(i, j, m)
                break
        # genCgate.append(l)
    genCgate = coo_matrix((data, (row, col)), shape=(NN, NN))
    return genCgate

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

def QuantuMeasure(N, psi, numMeas):
    # psi, N = initialize()
    # print(psi)
    # numMeas = int(input("How many measurements: "))
    # N = int(log2(len(psi)))
    # print(N)
    collapsedState = []
    # cnt = {}

    for _ in range(numMeas):
        r = random.random()
        q = abs(psi[0,0])**2
        
        for j in range(len(psi)): 
            # print(q)    
            if j == len(psi)-1:  
                # print(j)
                collapsedState.append('|'+f'{j:0{N}b}'+'>') #simplified using f string
                # print('|'+f'{j:0{N}b}'+'>\n')
                break
            elif r <= q:    
                # print(j)         
                collapsedState.append('|'+f'{j:0{N}b}'+'>')
                # print('|'+f'{j:0{N}b}'+">\n")
                break
            q += abs(psi[j+1, 0])**2
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
    p = gate.phase(0.5)
    print(gate.phase(0.5, 3, False))
    # print(coo_matrix(gate.Cgate(h)))
    # print(genCGate(2, h, 1, 2))
    # for i in h:
    #     for j in i:
    #         print(j , end=" ")
    #     print()
    # print(gate.CNOT[0,0])
    # N = gate.phase(theta = 1)
    # print(N)
    # M = gate.NOT
    # print(genCGate(3,M,2,1))
    # print(np.shape(H))
    # P = gate.phase(3,1)
    # print(P)
    # fullQuntumComp()
    # QuantuMeasure(psi, 10)
    # print(genCNOT(3,2,1))