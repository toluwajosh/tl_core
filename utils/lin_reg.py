import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D


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