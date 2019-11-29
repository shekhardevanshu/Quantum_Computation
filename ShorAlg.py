import random
import time
from math import sqrt, log2
import quantComp as qp
import numpy as np

def gcd(m,n):
    if m > n: m,n = n,m
    while True:
        if n%m == 0:
            return(m)
        else: m,n = n%m, m
def PerfectSquare(N):
    sq = sqrt(N)
    if int(sq + 0.1)**2 == N: 
        return(True)
    else: return(False)

def ClassicalShor(N):
    if N%2 == 0:
        return(2, N//2)
    l = list(range(2,N))
    while True:
        g = random.choice(l)
        if gcd(g,N) != 1:
            return(gcd(g,N), N // gcd(g,N))
        else:
            p = 1
            while(g**p % N != 1):
                p += 1
            if p % 2 == 0:
                x = gcd(N, g**(p//2) + 1)
                y = gcd(N, g**(p//2) - 1)
                if x*y == N: return(x,y)
def QunatumShor(C):
    if C%2 == 0:
        return(2, C//2)
    if PerfectSquare(C):
        return(int(sqrt(C)), int(sqrt(C)))
    # L,M = int(2*log2(C)), int(log2(C)) #L is qubits in x register and M is for f register
    L,M = 3,4
    N = L+M #total number of qubits

    #initialisation
    l1 = [0]*2**N
    l1[1] = 1
    psi = np.matrix(l1).transpose()
    l = list(range(2,C))
    while True:
        g = random.choice(l)
        # print(g)
        if gcd(g,N) != 1:
            return(gcd(g,N), N // gcd(g,N))
        else:
    #gate operations
            gate = qp.QGates(N)
            for i in range(L):
                # print(gate.Hadamard(L-i).shape, psi.shape)
                psi = np.matmul(gate.Hadamard(L-i), psi)
            D = np.zeros([2**N, 2**N])
            A = [(g**(2**i))%C for i in range(L)]
            # print(A)
            k = 0
            while(k<L):
                for i in range(2**L):
                    for j in range(2**L)   :
                        c = qp.int2bin(N,j)
                        cl, f = c[0:L], int(c[L:], 2)
                        if cl[L-1-k] == 0: D[i,j] = 1
                        elif cl[L-1-k] == 1 and f >= C: D[i,j] = 1
                        else:
                            # print(f)
                            f *= A[k]
                            
                            fb = qp.int2bin(N,f)
                            D[i, int(cl+fb,2)] = 1
                k += 1
            psi = np.matmul(D, psi)
            print(D)
        #QFT
            
            break


# print(gcd(18,4))
if __name__ == '__main__':
    t0 = time.time()
    # x = factor(314191)
    # x = ClassicalShor(63)
    # print(x)
    # print(time.time() - t0)
    # print(PerfectSquare(998001))
    QunatumShor(15)