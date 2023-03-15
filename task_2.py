import numpy as np


def check_convergence(A):
    for row in range(len(A)):
        if 2 * A[row][row] <= sum(A[row]):
            return False

    return True

def is_converged(diff, A_norm, eps):
    return np.linalg.norm(diff) <= eps \
                or A_norm > 1 / 2 and np.linalg.norm(diff) <= ((1 - A_norm) / A_norm) * eps

def solve_system(alpha, beta, eps):
    """
    Решает систему уравнений с помощью простого итерационного метода.

    Parameters:
        alpha (numpy.ndarray): Двумерный массив коэффициентов при неизвестных.
        beta (numpy.ndarray): Правая часть СЛАУ.
        eps (float): Критерий остановки.

    Returns:
        x (numpy.ndarray): Вектор решения системы уравнений.
    """
    # if not check_convergence(alpha):
    #     return None

    size = len(alpha)
    solution = [0] * size
    A_norm = np.linalg.norm(alpha)

    while True:

        next_step = [0] * size

        for row in range(size):
            sum = beta[row]

            for column in range(size):
                if column != row:
                    sum -= float(alpha[row][column]) * float(solution[column])

            next_step[row] = sum / float(alpha[row][row])

        diff = [0] * size
        for i in range(size):
            diff[i] = solution[i] - next_step[i]

        
        solution = next_step

        if is_converged(diff, A_norm, eps):
            return solution


alpha = np.array([
    [20, 2, 3, 7], 
    [1, 12, -2, -5],
    [5, -3, 13, 0], 
    [0, 0, -1, 15]
    ])
beta = np.array([5, 4, -3, 7])
x = solve_system(alpha, beta, 1e-3)
print(x)
