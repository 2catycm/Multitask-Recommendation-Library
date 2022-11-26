import numpy as np
from numpy.linalg import norm


def makenormal(u1, u2):
    u1 = u1 / norm(u1)
    u2 = u2 / norm(u2)
    cos_ = np.dot(u1, u2) / (norm(u1) * norm(u2))
    u_t = u1 * cos_
    u2 = u2 - u_t
    u2 = u2 / norm(u2)
    return u1, u2


if __name__ == "__main__":
    d = 10 # Feature number
    c = 1 # Scale constant
    p = 0.1 # Pearson correlation 
    datasize = 100000
    m = 0

    u1 = np.random.random(size=(d,))
    u2 = np.random.random(size=(d,))
    u1, u2 = makenormal(u1, u2)

    w1 = c * u1
    w2 = c * (p * u1 + np.sqrt(1 - p ** 2) * u2)

    alpha = np.random.random((m,))
    beta = np.random.random((m,))

    data = np.random.randn(datasize, d+2)

    out = open('./syn_data.csv', mode='w')
    for i in range(d+1):
        out.write(str(i)+",")
    out.write(str(d+1)+"\n")

    for da in data:
        x = da[0:d]
        e = np.random.normal(0, 0.01, 2)
        da[d] = np.dot(w1, x) + e[0]
        da[d+1] = np.dot(w2, x) + e[1]
        for i in range(m):
            da[d] += np.sin(alpha[i]*np.dot(w1, x)+beta[i])
            da[d+1] += np.sin(alpha[i]*np.dot(w2, x)+beta[i])
        for i in da[0:d+1]:
            out.write(str(i)+",")
        out.write(str(da[d+1])+"\n")
    out.close()
