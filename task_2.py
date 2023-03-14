import numpy as np

def check_convergence(A):
    size = len(A)

    for row in range(size):
            if A[row][row] <= sum(A[row]) - A[row][row]:
                return False

    return True

def solve_system(A, B, e):
    if not check_convergence(A): return None

    size = len(A)
    solution = [0] * size
    A_norm = np.linalg.norm(A)

    while True:
        solution_duplicate = [0] * size

        for row in range(size):
            sum = B[row]

            for column in range(size):
                if column != row: sum -= float(A[row][column]) * float(solution[column])


            solution_duplicate[row] = sum / float(A[row][row])

        diff = [0] * size
        for i in range(size): diff[i] = solution[i] - solution_duplicate[i]

        solution = solution_duplicate

        if np.linalg.norm(diff) <= e\
            or A_norm > 1/2 and np.linalg.norm(diff) <= ((1 - A_norm) /  A_norm) * e:
            return solution

alpha = np.array([[20, 2, 3, 7], [1, 12, -2, -5], [5, -3, 13, 0], [0, 0, -1, 15]])
beta = np.array([5, 4, -3, 7])
x = solve_system(alpha, beta, 1e-3)
print(x)
