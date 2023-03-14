import numpy as np


def gauss_elimination(A, B):
    n = len(A)
    # Гауссово исключение без поворота
    for i in range(n):
        # исключение
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i + 1, n):
                A[j][k] = A[j][k] - factor * A[i][k]
            B[j] = B[j] - factor * B[i]

    # Обратная замена
    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (B[i] - s) / A[i][i]

    return x


def gauss_elimination_with_pivoting(A, B):
    n = len(A)
    # Гауссово исключение с частичным поворотом
    for i in range(n):
        # Частичный поворот
        max_index = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[max_index][i]):
                max_index = j
        A[i], A[max_index] = A[max_index], A[i]
        B[i], B[max_index] = B[max_index], B[i]

        # исключение
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i + 1, n):
                A[j][k] = A[j][k] - factor * A[i][k]
            B[j] = B[j] - factor * B[i]

    # Обратная замена
    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (B[i] - s) / A[i][i]

    return x


'''
3x + 2y - z = 1
2x - 2y + 4z = -2
-x + 0.5y - z = 0
'''

A = [[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]]
B = [1, -2, 0]
x = gauss_elimination_with_pivoting(A, B)
print(x)