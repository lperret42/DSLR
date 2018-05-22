import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from src.math import logistic_function, logistic_cost, partial_derivative_n

class LogisticRegressor(object):
    def __init__(self, X, Y, learning_rate=1e3):
        self.X = X
        self.Y = Y
        self.ndims = len(X[0])
        self.theta = [0 for _ in range(self.ndims)]
        self.learning_rate = learning_rate
        self.nb_iter = 0

    def update_params(self):
        theta = [self.learning_rate * partial_derivative_n(self.theta, self.X, self.Y, n) for
                                    n in range(self.ndims)]
        self.theta = [self.theta[i] - theta[i] for i in range(self.ndims)]
        self.learning_rate *= 0.99

    def train(self, epsilon=1e-14, max_iter=float("inf"), show=False,
                                                        print_cost=False):
        last_cost = float("inf")
        print("ndims", self.ndims)
        while True:
            cost = logistic_cost(self.theta, self.X, self.Y)
            if print_cost:
                print(cost)
            if abs(last_cost - cost) <= epsilon:
                break
            self.update_params()
            last_cost = cost
            max_iter -= 1
            if max_iter <= 0:
                 break
            self.nb_iter += 1
        print(self.nb_iter)

    def save(self, print_cost=False):
        theta1_final = self.theta1 / self.sigma
        theta0_final = self.theta0 - (self.theta1 * self.mu) / self.sigma
        self.theta0 = theta0_final
        self.theta1 = theta1_final
        cost = get_cost(self.theta0, self.theta1, self.X, self.Y)
        if print_cost:
            print(cost, "\n")
        with open('params.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['theta0', 'theta1'])
            writer.writeheader()
            writer.writerow({'theta0': self.theta0, 'theta1':self.theta1})
            csvfile.close()
