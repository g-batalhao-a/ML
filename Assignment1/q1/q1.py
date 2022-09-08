# %%
from matplotlib.animation import FuncAnimation
import collections
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

x = np.genfromtxt(sys.argv[1]+'/X.csv', delimiter='\n')
y = np.genfromtxt(sys.argv[1]+'/Y.csv', delimiter='\n')

x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))

# %%
x.shape

# %%
y.shape

# %%
# Add intercept term to x
x = np.append(np.ones((x.shape)), x, axis=1)
# Normalize the data
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

# Num of examples and features (with intercept)
m = x.shape[0]
n = x.shape[1]

# %%
# Question 1.a - Batch gradient descent method for optimizing J(θ)

# Initialize params
theta = np.zeros((2, 1))
alpha = 1e-2

# Prediction function: h(θ) = x^Tθ


def predict(x, theta):
    return x.dot(theta)

# Cost function: J(θ) = 1/2m * Σ(y-h(θ))^2


def cost(x, y, theta):
    return (1/(2*m)) * np.sum((y - predict(x, theta))**2)


cost_0 = cost(x, y, theta)
print(f"Initial Cost value: {cost_0}")


def cost_grd(x, y, theta):
    return (1/m) * (np.zeros((2, 1)) + x.T.dot(x.dot(theta)-y))

# Gradient descent function


def batch_gradient_descent(x, y, theta, alpha, threshold=10e-8, num_iter=1000000):
    c = 0.0
    cost_hist = np.array([cost_0])
    theta_hist = theta
    i = 1
    while True:
        # Compute gradient and update theta
        theta -= alpha * cost_grd(x, y, theta)
        theta_hist = np.append(theta_hist, theta, axis=1)
        c = cost(x, y, theta)
        cost_hist = np.append(cost_hist, c)
        i += 1

        # Stop if the cost is below a threshold or if the num of iterations is above a certain amount
        if abs(cost_grd(x, y, theta)[1]) <= threshold or i >= num_iter:
            print(abs(cost_grd(x, y, theta)[1]))
            break

    return theta, cost_hist, theta_hist, i


theta, cost_hist, theta_hist, iterations = batch_gradient_descent(
    x, y, theta, alpha)

print(f'Cost of the model is {cost_hist[-1]} with {iterations} iterations')

# %%
# Question 1.b - pting data and hypothesis function

# Create predictions from our linear regression model and write them to a file
x_test = np.genfromtxt(sys.argv[2]+'/X.csv', delimiter='\n')
x_test = np.reshape(x_test, (-1, 1))
x_test = np.append(np.ones((x_test.shape)), x_test, axis=1)
x_test[:, 1] = (x_test[:, 1] - x_test[:, 1].mean()) / x_test[:, 1].std()
predictions = x_test.dot(theta)
with open('result_1.txt', 'w+') as f:
    for item in predictions:
        f.write('%s\n' % item[0])

    print("File written successfully")
f.close()