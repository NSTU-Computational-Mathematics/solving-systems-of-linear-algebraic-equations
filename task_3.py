import numpy as np


def solve_system(alpha, beta):
    size = len(alpha)
    solution = [0] * size

    for i in range(1, size):
        kf = alpha[i][i-1] / alpha[i-1][i-1]
        alpha[i][i] -= kf * alpha[i-1][i]
        beta[i] -= kf * beta[i-1]
        alpha[i][i - 1] = 0

    solution[size-1] = beta[size-1] / alpha[size-1][size-1]
    for i in range(size-2, -1, -1):
        solution[i] = (beta[i] - (solution[i+1] * alpha[i][i+1])) / alpha[i][i] 

    return solution


alpha = np.array([
    [8.0, -2, 0, 0, 0],
    [-1, 5, 3, 0, 0],
    [0, 7, -5, -9, 0],
    [0, 0, 4, 7, 9],
    [0, 0, 0, -5, 8]
])
beta = np.array([-7.0, 6, 9, -8, 5])
x = solve_system(alpha, beta)
print(x)
