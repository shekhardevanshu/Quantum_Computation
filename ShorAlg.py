import random
import time
from math import sqrt, log2, ceil, gcd
import quantComp as qp
import numpy as np
from fractions import Fraction
from statistics import mode
from scipy.sparse import coo_matrix, isspmatrix_dia

'''def gcd(m,n):
    if m > n: m,n = n,m
    while True:
        if n%m == 0:
            return(m)
        else: m,n = n%m, m'''


def PerfectSquare(N):
    sq = sqrt(N)
    if int(sq + 0.5)**2 == N:
        return(True)
    else:
        return(False)


def ClassicalShor(N):
    if N % 2 == 0:
        return(2, N // 2)
    l = list(range(2, N))
    while True:
        g = random.choice(l)
        if gcd(g, N) != 1:
            return(gcd(g, N), N // gcd(g, N))
        else:
            p = 1
            while(g**p % N != 1):
                p += 1
            if p % 2 == 0:
                x = gcd(N, g**(p // 2) + 1)
                y = gcd(N, g**(p // 2) - 1)
                if x * y == N:
                    return(x, y)


def intialize(C):
    print('initialize')
    if C % 2 == 0:
        return(2, C // 2)
    if PerfectSquare(C):
        return(int(sqrt(C)), int(sqrt(C)))
    # L is qubits in x register and M is for f register
    # L, M = ceil(2 * log2(C)), ceil(log2(C))
    # L, M = 3, 4  # for C = 15
    L, M = 6, 6  # for C = 39
    # print(L, M)
    N = L + M  # total number of qubits

    # initialisation
    psi = np.asmatrix(np.zeros((2**N, 1), dtype=float))
    psi[1, 0] = 1
    # psi = np.matrix(l1).transpose()
    return(psi, L, M, N)


# QFT
'''def QFT(N, L, i=1):
    # Remember: non-default argument(L) mustn't follow default arguments(i)
    print('qft')
    gate = qp.QGates(N)
    if i == L:
        return(gate.Hadamard(i))
    qft = gate.Hadamard(i)
    n = 1
    for j in range(i, L):
        n *= 0.5
        # print(n)
        phase = gate.phase(n)
        # print(phase)
        # print(j)
        qft = qft @ qp.CphaseGate(N, phase, i, j + 1)
    i += 1
    # print(i)
    return qft @ QFT(N, L, i)

# gate operations'''


def QFT(N, L):

    print('QFT')
    gate = qp.QGates(N)
    a = gate.Hadamard(L)
    for i in range(L-1, 0, -1):
        # print(i)
        qft = gate.Hadamard(i)
        n = 1
        for j in range(i, L):
            n *= 0.5
            phase = gate.phase(n)
            qft = qp.CphaseGate(N, phase, i, j + 1) @ qft
        a = a @ qft

    return a


def gateOperations(C, L, M, g):
    # print(C, g)
    print('gateOperations')
    N = L + M
    NN = 2**N
    MM = 2**M
    # LL = 2**L
    D = coo_matrix(np.eye(NN))
    # print(D)
    data = np.ones(NN)
    A = [(g**(2**i)) for i in range(L)]
    # print(A)
    k = 0
    while(k < L):
        row, col = [], []
        for j in range(NN):  # jth column

            f = j % MM
            y = j
            if (y >> (M + k)) & 1 == 0:
                i = j
                # D[i,j] = 1
                row.append(i)
                col.append(j)
                # data.append(1)

            elif (y >> (M + k)) & 1 == 1 and f >= C:
                i = j
                row.append(i)
                col.append(j)
            elif (y >> (M + k)) & 1 == 1 and f < C:
                # print(f)
                f *= A[k]
                f %= C
                cl = j // MM
                i = cl * MM + f
                # print(i, i1)
                row.append(i)
                col.append(j)
        d = coo_matrix((data, (row, col)), shape=(NN, NN))
        # print(d)
        # exit()
        D = D @ d
        k += 1

    return(D)


def measurement(N, psi, L, g, C):
    print('measurement')
    Mesured_states = qp.QuantuMeasure(N, psi, 100)
    # print(psi)
    X = []
    P = []
    F = []
    LL = 2**L
    for state in Mesured_states:
        # print(state)
        s = state[L:0:-1]
        # print(s)
        # print(state[L+1:-1])
        f = int(state[L+1:-1], 2)
        if f not in F:
            F.append(f)
        x = Fraction(int(s, 2) / LL)
        # print(int(s, 2))
        X.append(int(s, 2) / LL)
        P.append(x.denominator)
    cnt1, cnt2 = {}, {}
    # print(X)
    # print(F)
    for i, j in zip(X, P):
        cnt1[i] = cnt1.get(i, 0) + 1
        cnt2[j] = cnt2.get(j, 0) + 1
    '''i = 2
    while i <= 5:
        print(i)
        for p in P:
            if (g ** p) % C == 1:
                return p
        P = list(map(lambda x: i*x, P))
        i += 1'''

    # print(P)
    print(cnt1)
    print(cnt2)
    exit()
    # print(cnt2)
    # try:
    # p = mode(P)
    # except:
    # measurement(N, psi, L)
    '''for k in F:
        # print(mode(P))
        cnt2 = {}
        if mode(P) in F:
            return(mode(P))
        else:
            if k == 1:
                continue
            for i, j in enumerate(P):
                # print(i, j)
                if j != 1 and j < k and k % j == 0:
                    # print(k)
                    P[i] = k
        for h in P:
            cnt2[h] = cnt2.get(h, 0) + 1
        print(cnt2)'''
    # if mode(P) in F:
    # return(mode(p))


'''def frac(x):
    s = str(x)
    n = len(s)-s.find('.')-1
    x = ceil(x*10**n)
    N, D = x, 10**n
    g = gcd(N, D)
    while g != 1:
        N //= g
        D //= g
        g = gcd(N, D)
    return(str(int(N))+'/'+str(int(D)))'''


def QunatumShor(C):
    psi, L, M, N = intialize(C)
    # psi = coo_matrix(psi)
    # print(psi.todense())
    # exit()
    gate = qp.QGates(N)
    for i in range(L):
        psi = gate.Hadamard(L - i) @ psi
    # print(psi)
    # exit()
    while True:
        # l = list(range(2, C))
        # g = random.choice(l)
        g = 10
        # print(g)
        if gcd(g, C) != 1:
            print('correct guess')
            return(gcd(g, C), C // gcd(g, C))
        D = gateOperations(C, L, M, g)
        # print(D.todense())
        # exit()
        psi = D @ psi
        # print(psi)
        # exit()8
        qft = QFT(N, L)
        psi = qft @ psi
        # print(psi)
        # exit()
        p = measurement(N, psi, L, g, C)
        if type(p) != int:
            continue
        print(p)
        # print(g**p % C)
        # exit()
        if p % 2 == 0:
            x = gcd(C, g ** (p // 2) + 1)
            y = gcd(C, g ** (p // 2) - 1)
            print(x, y)
            if x * y == C and (x != 1 and y != 1):
                return(x, y)


# print(gcd(18,4))
if __name__ == '__main__':
    # pass
    # t0 = time.time()
    # x = factor(314191)

    # x = ClassicalShor(15)
    # print(x)
    # print(time.time() - t0)

    # t0 = time.time()
    factors = QunatumShor(39)
    print(factors)
    # print(time.time() - t0)
    # print(frac(0.623))
    # print(Fraction(0.58).limit_denominator())
    # gateOperations(39,11,6,15)

    # D = gateOperations(33, 11, 6, 5)
    # print(D)
