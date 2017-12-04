import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D

# from the tutorial on: https://mubaris.com/2017-09-28/linear-regression-from-scratch
# the code here is incomplete

X = np.array([x0, math, read]).T
B = np.array([0, 0, 0])
Y = np.array(write) # dependent variable
alpha = 0.0001


def cost_function(X, Y, B):
  m = len(Y)
  J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
  return J

initial_cost = cost_function(X, Y, B)

def gradient_descent(X, Y, B, alpha, iterations):
  cost_history = [0] * iterations
  m = len(Y)
  
  for iteration in range(iterations):
    # Hypothesis Values
    h = X.dot(B)
    # Difference b/w Hypothesis and Actual Y
    loss = h - Y
    # Gradient Calculation
    gradient = X.T.dot(loss) / m
    # Changing Values of B using Gradient
    B = B - alpha * gradient
    # New Cost Value
    cost = cost_function(X, Y, B)
    cost_history[iteration] = cost
    
  return B, cost_history


# 100000 Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)


# New Values of B
print(newB)

# Final Cost of new B
print(cost_history[-1])

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Y_pred = X.dot(newB)

print(rmse(Y, Y_pred))
print(r2_score(Y, Y_pred))