import numpy as np

def check_convergence(A):
    size = len(A)

    for row in range(size):
        for column in range(size):
            if 2 * A[row][column] <= sum(A[row]):
                return False

    return True

def solve_system(A, B, e):
    size = len(A)
    solution = [0] * size

    while True:
        solution_duplicate = [0] * size
        for k in range(size):
            solution_duplicate[k] = (B[k] - sum((A[i] * solution[i] if i != k else 0) for i in range(size))) / A[k][k]

        diff = [0] * size
        for i in range(size): diff[i] = solution[i] - solution_duplicate[i]

        solution = solution_duplicate

        A_norm = np.linalg.norm(A)
        if np.linalg.norm(diff) <= ((1 - A_norm) /  A_norm) * e:
            return solution

alpha = np.array([[1, 9, 1], [2, 2, 11], [10, 2, 1]])
beta = np.array([12, 26, 14])
x = solve_system(alpha, beta, 1e-3)
print(x)
# import numpy as np
#
#
# def solve_system(alpha, beta, epsilon, check_convergence=True):
#     """
#     Решает систему уравнений с помощью простого итерационного метода.
#
#     Parameters:
#         alpha (numpy.ndarray): Двумерный массив значений.
#         beta (numpy.ndarray): Вертикальная решетка.
#         epsilon (float): Критерий остановки.
#         check_convergence (bool): Необязательный параметр для проверки условия сходимости. По умолчанию имеет значение True.
#
#     Returns:
#         x (numpy.ndarray): Решение системы уравнений.
#     """
#
#     # Transform the system of equations for convergence
#     alpha_norm = np.linalg.norm(alpha)
#     if check_convergence and alpha_norm >= 1:
#         alpha = alpha / alpha_norm ** 2
#         beta = beta / alpha_norm
#
#     # Initialize x
#     x_prev = np.zeros_like(beta)
#     x = np.copy(beta)
#
#     # Iterate until stopping criterion is met
#     while np.linalg.norm(x - x_prev) > epsilon:
#         x_prev = np.copy(x)
#         x = np.dot(alpha, x) + beta
#
#     # Add an ending check to ensure convergence
#     if check_convergence:
#         delta = (1 - alpha_norm) / alpha_norm * epsilon
#         if np.linalg.norm(x - x_prev) > delta:
#             raise ValueError("Итерационный процесс не сходится.")
#
#     return x
#
#
# # Example usage
# alpha = np.array([[1, 9, 1], [2, 2, 11], [10, 2, 1]])
# beta = np.array([12, 26, 14])
# x = solve_system(alpha, beta, epsilon=1e-3)
# print(x)
