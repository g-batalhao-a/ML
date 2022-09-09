# %%
# Question 1: Gaussian Discrmimant Analysis
import sys
import math
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

x = np.genfromtxt(sys.argv[1]+'/X.csv', delimiter=',')
y = np.genfromtxt(sys.argv[1]+'/Y.csv', delimiter=',', dtype=str)

y = np.reshape(y, (-1, 1))
# Num of examples
m = x.shape[0]
# %%
# Normalize the data
x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()


# %%
# GDA params
y[y == 'Alaska'] = 0
y[y == 'Canada'] = 1
phi = np.sum([y == '0'])/m

i_x0 = [i for i in range(len(y)) if y[i] == '0']
i_x1 = [i for i in range(len(y)) if y[i] == '1']
m_x0 = (np.mean(x[i_x0], axis=0)).reshape(1, -1)
m_x1 = (np.mean(x[i_x1], axis=0)).reshape(1, -1)


def covariance(x):
    c = np.zeros((2, 2))
    c0 = np.zeros((2, 2))
    c1 = np.zeros((2, 2))
    m = len(x)
    c0 = np.cov(x[i_x0], rowvar=False)
    c1 = np.cov(x[i_x1], rowvar=False)
    c = (c0+c1)/m

    return c, c0, c1


cov, cov_0, cov_1 = covariance(x)
print(f"Mean of distribution of Alaska: {m_x0} and covariance: {cov_0}")
print(f"Mean of distribution Canada: {m_x1} and covariance: {cov_1}")
print(f"Covariance matrix: {cov}")

# −1/2(x−μc)⊤Σ−1c(x−μc)−12log|Σc|+logπc


def prob_class(x, mu, cov):
    normal_distribution_prob = multivariate_normal(mean=mu, cov=cov)
    return np.log(phi) + normal_distribution_prob.logpdf(x)


def predict(x, m_x0, m_x1, cov_0, cov_1):
    y_pred = []
    for i in range(x.shape[0]):
        print(prob_class(x[i:i+1], m_x0, cov_0),
              prob_class(x[i:i+1], m_x1, cov_1))
        if prob_class(x[i:i+1], m_x0, cov_0) >= prob_class(x[i:i+1], m_x1, cov_1):
            y_pred = np.append(y_pred, "Alaska")
        else:
            y_pred = np.append(y_pred, "Canada")
    return y_pred


x_test = np.genfromtxt(sys.argv[2]+'/X.csv', delimiter=',')
x_test[:, 0] = (x_test[:, 0] - x_test[:, 0].mean()) / x_test[:, 0].std()
x_test[:, 1] = (x_test[:, 1] - x_test[:, 1].mean()) / x_test[:, 1].std()
predictions = predict(x_test, m_x0.ravel(), m_x1.ravel(), cov_0, cov_1)
with open('result_4.txt', 'w+') as f:
    for item in predictions:
        f.write("%s\n" % item)

    print("File written successfully")
