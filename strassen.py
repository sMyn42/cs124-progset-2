import numpy as np
import math
import random
import sys





def main():
    arrfile = sys.argv[1]
    f = open(arrfile, 'r')
    entries = []
    for line in f:
        entries.append(int(line.strip()))
    ds = int(len(entries)/2)
    d = int(math.sqrt(ds))
    a = np.reshape(np.array(entries[:ds], dtype=np.int32), (d, d))
    b = np.reshape(np.array(entries[ds:], dtype=np.int32), (d, d))
    print(mat_mul(a, b))
    return 0

def strassen_variant(arr):
    return 0
    

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
    return 0
    
def time_algos():
    return 0

if __name__ == "__main__":
    #main()
    time_algos()