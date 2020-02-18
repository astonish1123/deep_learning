import numpy as np

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    ce = -1*np.sum(Y*np.log(P)+(1-Y)*np.log(1-P))
    return ce

if __name__ == '__main__':
    Y = [1,0,1,1]
    P = [0.4, 0.6, 0.1, 0.5]

    print(cross_entropy(Y, P))
    