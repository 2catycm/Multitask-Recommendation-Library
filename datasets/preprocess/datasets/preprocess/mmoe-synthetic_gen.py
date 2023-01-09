import argparse
import pandas as pd
import numpy as np
def get_data(datasize,feature_num,scale,correlation,m):
    feature_num = 100
    datasize = 50000
    scale = 0.3
    correlation = 0.8
    # Initialize vectors u1, u2, w1, and w2 according to the paper
    mu1 = np.random.normal(size=feature_num)
    mu1 = (mu1 - np.mean(mu1)) / (np.std(mu1) * np.sqrt(feature_num))
    mu2 = np.random.normal(size=feature_num)
    mu2 -= mu2.dot(mu1) * mu1
    mu2 /= np.linalg.norm(mu2)
    w1 = scale * mu1
    w2 = scale * (correlation * mu1 + np.sqrt(1. - correlation ** 2) * mu2)

    # Feature and label generation
    alpha = np.random.normal(size=m)
    beta = np.random.normal(size=m)
    y0 = []
    y1 = []
    X = []

    for i in range(datasize):
        x = np.random.normal(size=feature_num)
        X.append(x)
        num1 = w1.dot(x)
        num2 = w2.dot(x)
        comp1, comp2 = 0.0, 0.0

        for j in range(m):
            comp1 += np.sin(alpha[j] * num1 + beta[j])
            comp2 += np.sin(alpha[j] * num2 + beta[j])

        y0.append(num1 + comp1 + np.random.normal(scale=0.1, size=1))
        y1.append(num2 + comp2 + np.random.normal(scale=0.1, size=1))

    X = np.array(X)
    data = pd.DataFrame(
        data=X,
        index=range(X.shape[0]),
        columns=['x{}'.format(it) for it in range(X.shape[1])]
    )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasize")
    parser.add_argument("feature_num")
    parser.add_argument("scale")
    parser.add_argument("correlation")
    parser.add_argument("m")
    args = parser.parse_args()
    data = get_data(args.datasize,args.feature_num,args.scale,args.correlation,args.m)
    data.to_csv(args.path,sep=',',index=False,header=True)
    
if __name__ == '__main__':
    main()
