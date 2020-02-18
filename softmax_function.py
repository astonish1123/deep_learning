import numpy as np

def softmax(L):
    e = np.exp(L)
    sum_e = np.sum(e)
    outcomes = []
    for i in e:
        outcomes.append(i*1/sum_e)
    return outcomes

if __name__ == '__main__':
    L = [5,6,7]
    results = softmax(L)
    print(results)
