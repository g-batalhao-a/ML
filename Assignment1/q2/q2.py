# %%
# Question 2: Logistic Regression
from matplotlib.animation import FuncAnimation
import time
import sys
import numpy as np
import pandas as pd

dataset = np.genfromtxt(sys.argv[1]+'/X.csv', delimiter=',')

x1 = dataset[:, 0].reshape(-1, 1)
x2 = dataset[:, 1].reshape(-1, 1)
x0 = np.ones((x1.shape))
x_test = np.append(x0, x1, axis=1)
x_test = np.append(x_test, x2, axis=1)

n = x_test.shape[0]

# %%
# Question 2.a - Sample million of points

# Data points given
data_points = 1000000
# Intercept term
x0 = np.ones((data_points, 1))

# Samples
x1 = np.random.normal(3, 2, data_points).reshape(-1, 1)
x2 = np.random.normal(-1, 2, data_points).reshape(-1, 1)

# Create matrix
x = np.append(x0, x1, axis=1)
x = np.append(x, x2, axis=1)

# Sample epsilon error
eps = np.random.normal(0, np.sqrt(2), data_points).reshape(-1, 1)

theta = np.array([[3], [1], [2]])

# Generate the value of Y (given X, parameterized by given Theta)
y = x.dot(theta) + eps

# Shuffle data
temp = np.append(x, y, axis=1)
np.random.shuffle(temp)

x = temp[:, 0:3]
y = temp[:, -1:]

# %%
# Question 2.b - Apply Stochastic gradient descent

alpha = 0.001

# Prediction function: h(θ) = x^Tθ


def predict(x, theta):
    return x.dot(theta)

# Cost function: J(θ) = 1/2m * Σ(y-h(θ))^2


def cost(x, y, theta, m):
    return (1/(2*m)) * np.sum((y - predict(x, theta))**2)


cost_0 = cost(x, y, theta, data_points)
print(f"Initial Cost:{cost_0}")


def cost_grd(x, y, theta, m):
    return (1/m) * (np.zeros((3, 1)) + x.T.dot(x.dot(theta)-y))


# Stochastic descent function
def stochastic_gradient_descent(x, y, theta, alpha, r, batches, threshold=10e-7):

    start = time.time()
    theta = np.zeros((3, 1))
    i = 0
    c, c_avg = 0.0, np.array([cost(x[0], y[0], theta, 1)])
    theta_hist = theta

    while True:
        i += 1
        count = 0
        c_init = cost(x, y, theta, data_points)

        for b in batches:
            c += cost(b[0], b[1], theta, r)

            check = 10000 if r == 1 else 100 if r == 100 else 10 if r == 10000 else 1
            if (count % check == 0 and count != 0):
                c /= check
                c_avg = np.append(c_avg, c)
                c = 0.0

            theta -= alpha * cost_grd(b[0], b[1], theta, r)
            theta_hist = np.append(theta_hist, theta, axis=1)
            count += 1

        c_final = cost(x, y, theta, data_points)
        if (abs(c_final - c_init) < threshold):
            break

    end = time.time()

    return theta, c_final, theta_hist, i, end-start


batches = [(x[i:i+10000, :], y[i:i+10000]) for i in range(0, data_points, 10000)]
theta, c_final, theta_hist, iterations, t = stochastic_gradient_descent(
    x, y, theta, alpha, 10000, batches)
print(
    f'Cost of the model is {c_final} with {iterations} iterations and {t} s')

predictions = predict(x_test, theta)
with open('result_2.txt', 'w+') as f:
    for item in predictions:
        f.write('%s\n' % item[0])

    print("File written successfully")
f.close()
