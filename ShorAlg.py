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
    L, M = 3, 4
    # print(L, M)
    N = L + M  # total number of qubits

    # initialisation
    psi = np.asmatrix(np.zeros((2**N, 1), dtype=float))
    psi[1, 0] = 1
    # psi = np.matrix(l1).transpose()
    return(psi, L, M, N)


# QFT
def QFT(N, L, i=1):
    # Remember: non-default argument(L) mustn't follow default arguments(i)
    print('qft')
    gate = qp.QGates(N)
    if i == L:
        return(gate.Hadamard(i))
    qft = gate.Hadamard(i)
    for j in range(i, L):
        n = (1 / 2)**j
        phase = gate.phase(n)
        # print(phase)
        # print(j)
        qft = qft @ qp.genCGate(N, phase, i, j + 1)
    i += 1
    # print(i)
    return qft @ QFT(N, L, i)

# gate operations


def QFTItr(N, L):
    print('QFTItr')
    gate = qp.QGates(N)
    a = gate.Hadamard(L)
    for i in range(1, L):
        print(i)
        qft = gate.Hadamard(i)
        n = 1
        for j in range(i, L):
            n *= 0.5
            phase = gate.phase(n)
            qft = qft @ qp.genCGate(N, phase, i, j + 1)
        a = qft @ a

    return a


def gateOperations(C, L, M, g):
    print('gateOperations')
    N = L + M
    NN = 2**N
    MM = 2**M
    # LL = 2**L
    # ds = []
    # D = np.zeros([2**N, 2**N])
    D = coo_matrix(np.eye(NN))
    # print(D)
    data = np.ones(NN)
    A = [(g**(2**i)) for i in range(L)]
    # print(A)
    k = 0
    while(k < L):
        row, col = [], []
        # print(M+k)
        for j in range(NN):  # jth column
            # c = qp.int2bin(N,j)
            # cl, f1 = c[0:L], int(c[L:], 2)
            # print(c, j, f)

            f = j % MM
            y = j
            # print(f, f1)
            # ds.append((j >> (M + k)) & 1)
            if (y >> (M + k)) & 1 == 0:
                i = j
                # D[i,j] = 1
                row.append(i)
                col.append(j)
                # data.append(1)

            elif (y >> (M + k)) & 1 == 1 and f >= C:
                # print(f)
                i = j
                # D[i,j] = 1
                row.append(i)
                col.append(j)
                # data.append(1)
                # break
            elif (y >> (M + k)) & 1 == 1 and f < C:
                # print(f)
                f *= A[k]
                f %= C
                # print(f)
                # fb = qp.int2bin(M,f)
                # cl = ''
                # for v in range(L):
                # c = cl
                # cl = str((j >> M+v) & 1) + c
                # i = int(cl + fb, 2)
                cl = j // MM
                i = cl * MM + f
                # print(i, i1)
                row.append(i)
                col.append(j)
                # data.append(1)
                # print(i, cl+fb)
                # D[i, j] = 1
                # print(i,D[i,j])
                # break
        # print(len(row), len(col), len(data))
        # print(D.shape)
        # print(ds)
        d = coo_matrix((data, (row, col)), shape=(NN, NN))
        # print(d)
        # exit()
        D = D @ d
        k += 1

    return(D)

    # print(psi)
    # print(D @ D.transpose())
    # print(np.matmul(QFT(L), psi))
    # print(qp.QuantuMeasure(psi, 10))


def measurement(N, psi, L):
    print('measurement')
    Mesured_states = qp.QuantuMeasure(N, psi, 10)
    # print(psi)
    X = []
    P = []
    for state in Mesured_states:
        # print(state)
        s = state[L:0:-1]
        # print(s)
        x = Fraction(int(s, 2) / 2**L).limit_denominator()
        # print(x)
        X.append(int(s, 2) / 2**L)
        P.append(x.denominator)
    cnt = {}
    # print(X)
    for i in X:
        cnt[i] = cnt.get(i, 0) + 1
    # print(P)
    # print(cnt)
    try:
        p = mode(P)
        return(p)
    except:
        measurement(N, psi, L)


'''def frac(x):
    s = str(x)
    n = len(s)-s.find('.')-1
    x = ceil(x*10**n)
    N, D = x, 10**n
    g = gcd(N, D)
    while g != 1:
        N /= g
        D /= g
        g = gcd(N, D)
    return(str(int(N))+'/'+str(int(D)))'''


def QunatumShor(C):
    psi, L, M, N = intialize(C)
    psi = coo_matrix(psi)
    # print(psi)
    gate = qp.QGates(N)
    for i in range(L):
        psi = gate.Hadamard(L - i) @ psi
    # print(psi)
    # exit()
    # l = list(range(2, C))
    # g = random.choice(l)
    g = 7
    # print(g)
    if gcd(g, C) != 1:
        return(gcd(g, C), C // gcd(g, C))
    D = gateOperations(C, L, M, g)
    # print(D.todense())
    # exit()
    psi = D @ psi
    # print(coo_matrix(psi))
    qft = QFTItr(N, L)
    psi = qft @ psi
    # print(len(psi))
    # exit()
    while True:
        p = measurement(N, psi.todense(), L)
        print(p)
        if g**p % C == 1 and p % 2 == 0:
            x = gcd(C, g**(p // 2) + 1)
            y = gcd(C, g**(p // 2) - 1)
            if x * y == C and (x != 1 and y != 1):
                return(x, y)
        else:
            QunatumShor(C)


# print(gcd(18,4))
if __name__ == '__main__':
    # pass
    # t0 = time.time()
    # x = factor(314191)

    # x = ClassicalShor(15)
    # print(x)
    # print(time.time() - t0)

    # t0 = time.time()
    factors = QunatumShor(15)
    print(factors)
    # print(time.time() - t0)
    # print(frac(0.58))
    # print(Fraction(0.58).limit_denominator())
    # gateOperations(39,11,6,15)

    # D = gateOperations(33, 11, 6, 5)
    # print(D)
