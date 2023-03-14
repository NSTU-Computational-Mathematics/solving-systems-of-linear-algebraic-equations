import numpy as np

def check_convergence(A):
    size = len(A)

    for row in range(size):
        for column in range(size):
            if 2 * A[row][column] <= sum(A[row]):
                return False

    return True

def solve_system(A, B, e):
    # if not check_convergence(A): return None

    size = len(A)
    solution = [0] * size

    while True:
        solution_duplicate = [0] * size

        for row in range(size):
            sum = B[row]
            for column in range(size):
                if column != row: sum -= A[row][column]

            solution_duplicate[row] = sum / A[row][row]

        diff = [0] * size
        for i in range(size): diff[i] = solution[i] - solution_duplicate[i]

        solution = solution_duplicate

        A_norm = np.linalg.norm(A)
        if A_norm > 1/2 and np.linalg.norm(diff) <= ((1 - A_norm) /  A_norm) * e:
            return solution_duplicate
        elif np.linalg.norm(diff) <= e:
            return solution_duplicate


alpha = np.array([[1, 9, 1], [2, 2, 11], [10, 2, 1]])
beta = np.array([12, 26, 14])
x = solve_system(alpha, beta, 1e-3)
print(x)
