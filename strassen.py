import numpy as np
import math
import random
import sys

def main():
    arrfile = sys.argv[3]
    f = open(arrfile, 'r')
    entries = []
    for line in f:
        entries.append(int(line.strip()))
    ds = int(len(entries)/2)
    d = int(math.sqrt(ds))
    a = np.reshape(np.array(entries[:ds], dtype=np.int32), (d, d))
    b = np.reshape(np.array(entries[ds:], dtype=np.int32), (d, d))
    c = strassen_variant(a, b, 64)
    output_diag(c)
    f.close()
    return 0

# apply strassen's algorithm recursively with padding
def strassen_variant(a, b, n0):
    n = a.shape[0]

    # the sum of the two below quantities should always add to n.

    ns = math.ceil(n/2)
    nl = math.floor(n/2)
    if n <= n0:
        return mat_mul(a, b)
    
    # define submatrices with padding if needed. Padding is removed by
    # using [0:n, 0:n] on intermediate arrays before addition.
    # odd matrices

    if n % 2 == 1:
        a = np.pad(a, ((0, 1), (0, 1)), constant_values=0)
        b = np.pad(b, ((0, 1), (0, 1)), constant_values=0)
    
    # Capital letters used for readability/alignment with lecture notes.

    A = a[:ns, :ns]
    B = a[:ns, ns:]
    C = a[ns:, :ns]
    D = a[ns:, ns:]

    E = b[:ns, :ns]
    F = b[:ns, ns:]
    G = b[ns:, :ns]
    H = b[ns:, ns:]

    # recursive calls:

    p1 = strassen_variant(A, np.subtract(F, H), n0)
    p2 = strassen_variant(np.add(A, B), H, n0)
    p3 = strassen_variant(np.add(C, D), E, n0)
    p4 = strassen_variant(D, np.subtract(G, E), n0)
    p5 = strassen_variant(np.add(A, D), np.add(E, H), n0)
    p6 = strassen_variant(np.subtract(B, D), np.add(G, H), n0)
    p7 = strassen_variant(np.subtract(C, A), np.add(E, F), n0)

    # recombination

    c1 = (p4 + p5 + p6 - p2)
    c2 = (p1 + p2)[:, :n-ns]
    c3 = (p3 + p4)[:n-ns, :]
    c4 = (p1 - p3 + p5 + p7)[:n-ns, :n-ns]

    c = np.vstack((np.hstack((c1, c2)), np.hstack((c3, c4))))

    return c
    

def mat_mul(a, b):
    c = np.zeros(a.shape, dtype=np.int32)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            c[i, j] = int(np.dot(a[i, :], b[:, j]))
    return c
    
def create_test_matrix(n, l):
    p = len(l)
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a[i, j] = math.floor(random.random() * p)     
    return a

def output_diag(arr):
    for i in range(arr.shape[0]):
        print(arr[i, i])
    return 0
    
def time_algos():
    return 0

def find_triangles():
    return 0

if __name__ == "__main__":
    main()
    #time_algos()