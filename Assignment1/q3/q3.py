# %%
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

x = np.genfromtxt(sys.argv[1]+'/X.csv', delimiter=',')
y = np.genfromtxt(sys.argv[1]+'/Y.csv', delimiter='\n')

y = np.reshape(y,(-1,1))

# %%
# Normalize the data
x = (x - x.mean()) / x.std()
# Add intercept term to x
x = np.append(np.ones((x.shape[0],1)),x,axis=1)

# Num of examples and features (with intercept)
m = x.shape[0]
n = x.shape[1]


# %%
# Question 3.a - Implement Newton’s method

# Initialize params
theta = np.zeros((n, 1))

# H(LL) = x.T*diag(o(x.theta)(1-o(x.theta)))*x
d = x.dot(theta)
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
sig = sigmoid(d)
diag = np.identity(x.shape[0]) * sig.T.dot(1-sig)
H = x.T.dot(diag.dot(x))

#Newton's Update Equation
grad = x.T.dot((sig-y))
t_final= theta - np.linalg.inv(H).dot(grad)
print(f"Theta from Newton's Method: {t_final}")

def predict(x, theta):
    return sigmoid(x.dot(theta))

x_test = np.genfromtxt(sys.argv[2]+'/X.csv', delimiter=',')
x_test = (x_test- x_test.mean()) / x_test.std()
x_test = np.append(np.ones((x_test.shape[0],1)),x_test,axis=1)
predictions = predict(x_test, t_final)
with open('result_3.txt', 'w+') as f:
    for item in predictions:
        if item[0] >= 0.5:
            f.write('1\n')
        else:
            f.write('0\n')
     
    print("File written successfully")
