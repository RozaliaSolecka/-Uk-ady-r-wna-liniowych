from methods import *
import time
from matplotlib import pyplot

####B
def jacobi(matrix_A, vector_b, N):
    previous_vector_x = ones(N)
    number_of_iterations = 0
    res = vector_sub_vector(vector_multiply_matrix(previous_vector_x, matrix_A), vector_b)
    time1 = time.time()

    while norm(res) > 10 ** -9:
        actual_vector_x = []

        for i in range(N):
            b = vector_b[i]
            for j in range(N):
                if i != j:
                    b -= matrix_A[i][j]*previous_vector_x[j]
            b /= matrix_A[i][i]
            actual_vector_x.append(b)
        previous_vector_x = copy_vector(actual_vector_x)
        res = vector_sub_vector(vector_multiply_matrix(previous_vector_x, matrix_A), vector_b)
        number_of_iterations = number_of_iterations + 1

    time2 = time.time()
    print("Jacobi's method")
    print('time:', time2 - time1)
    print('iterations:', number_of_iterations)
    print('residuum norm', norm(res))
    print()
    return time2 - time1


def gaussseidel(matrix_A, vector_b, N):
    actual_vector_x = ones(N)
    previous_vector_x = ones(N)
    number_of_iterations = 0
    res = vector_sub_vector(vector_multiply_matrix(previous_vector_x, matrix_A), vector_b)
    time1 = time.time()

    while norm(res) > 10 ** -9:
        for i in range(N):
            b = vector_b[i]
            for j in range(N):
                if i > j:
                    b -= matrix_A[i][j]*actual_vector_x[j]
                elif i < j:
                     b -= matrix_A[i][j] * previous_vector_x[j]
            b /= matrix_A[i][i]
            actual_vector_x[i] = b
        previous_vector_x = copy_vector(actual_vector_x)
        res = vector_sub_vector(vector_multiply_matrix(actual_vector_x, matrix_A), vector_b)
        number_of_iterations = number_of_iterations + 1

    time2 = time.time()
    print("Gauss-Seidel's method")
    print('time:', time2 - time1)
    print('iterations:', number_of_iterations)
    print('residuum norm', norm(res))
    print()
    return time2 - time1



####C
def jacobi2(matrix_A, vector_b, N):
    previous_vector_x = ones(N)
    number_of_iterations = 0
    res = vector_sub_vector(vector_multiply_matrix(previous_vector_x, matrix_A), vector_b)
    normres = []
    time1 = time.time()

    while norm(res) > 10 ** -9:
        actual_vector_x = []
        for i in range(N):
            b = vector_b[i]
            for j in range(N):
                if i != j:
                    b -= matrix_A[i][j]*previous_vector_x[j]
            b /= matrix_A[i][i]
            actual_vector_x.append(b)
        previous_vector_x = copy_vector(actual_vector_x)
        res = vector_sub_vector(vector_multiply_matrix(previous_vector_x, matrix_A), vector_b)
        number_of_iterations = number_of_iterations + 1
        normres.append(norm(res))
        print(number_of_iterations)
        print(norm(res))

        if number_of_iterations > 1200:
            break

    time2 = time.time()
    print("Jacobi's method")
    print('time:', time2 - time1)
    print('iterations:', number_of_iterations)
    print()

    pyplot.plot(normres)
    pyplot.title("Jacobi's method")
    pyplot.xlabel('number of iterations')
    pyplot.ylabel('norm(res)')
    pyplot.show()

    return time2 - time1


def gaussseidel2(matrix_A, vector_b, N):
    actual_vector_x = ones(N)
    previous_vector_x = ones(N)
    number_of_iterations = 0
    res = vector_sub_vector(vector_multiply_matrix(previous_vector_x, matrix_A), vector_b)
    normres = []
    time1 = time.time()

    while norm(res) > 10 ** -9:
        for i in range(N):
            b = vector_b[i]
            for j in range(N):
                if i > j:
                    b -= matrix_A[i][j]*actual_vector_x[j]
                elif i < j:
                    b -= matrix_A[i][j] * previous_vector_x[j]
            b /= matrix_A[i][i]
            actual_vector_x[i] = b
        previous_vector_x = copy_vector(actual_vector_x)
        res = vector_sub_vector(vector_multiply_matrix(actual_vector_x, matrix_A), vector_b)
        number_of_iterations = number_of_iterations + 1
        normres.append(norm(res))
        print(number_of_iterations)
        print(norm(res))

        if number_of_iterations > 500:
            break

    time2= time.time()
    print("Gauss-Seidel's method")
    print('time:', time2 - time1)
    print('iterations:', number_of_iterations)
    print()

    pyplot.plot(normres)
    pyplot.title("Gauss Seidel's method")
    pyplot.xlabel('number of iterations')
    pyplot.ylabel('norm(res)')
    pyplot.show()

    return time2 - time1

####D

def lu_decomposition(matrix, vector_b, N):
    vector_y = ones(N)
    vector_x = ones(N)
    matrix_L = identity_matrix(N)
    matrix_U = copy_matrix(matrix)

# TWORZENIE MACIERZY L I U

    time1 = time.time()

    for k in range(N-1):
        for j in range(k+1, N):
            matrix_L[j][k] = matrix_U[j][k] / matrix_U[k][k]
            for i in range(k,N):
                matrix_U[j][i] = matrix_U[j][i] - (matrix_L[j][k] * matrix_U[k][i])

    #if matrix_sub_matrix(matrix, matrix_multiply_matrix(matrix_L, matrix_U, N), N) == zero_matrix(N):

    # Ly = b
    for i in range(N):
         sum = 0
         for k in range(i):
            sum += matrix_L[i][k] * vector_y[k]
         vector_y[i] = (1 / matrix_L[i][i])*(vector_b[i] - sum)

    # Ux = y
    for i in range(N - 1, -1, -1):
        sum = 0
        for k in range(N - 1, i, -1):
            sum += matrix_U[i][k] * vector_x[k]
        vector_x[i] = (1 / matrix_U[i][i]) * (vector_y[i] - sum)

    time2 = time.time()
    res = vector_sub_vector(vector_multiply_matrix(vector_x, matrix), vector_b)


    print("LU method")
    print("residuum norm:", norm(res))
    print('time:', time2 - time1)

    return time2 - time1


if __name__ == "__main__":

    # DANE
    N = 935

    # ZADANIE A
    matrix_A = create_matrix_A(N)
    vector_b = create_vector_b(N)

    # ZADANIE B
    jacobi(matrix_A, vector_b, len(matrix_A))
    gaussseidel(matrix_A, vector_b, len(matrix_A))

    # ZADANIE C
    matrix_C = create_matrix_C(N)
    jacobi2(matrix_C, vector_b, len(matrix_C))
    gaussseidel2(matrix_C, vector_b, len(matrix_C))

    # ZADANIE D
    lu_decomposition(matrix_C, vector_b, len(matrix_C))

    # ZADANIE E
    NUMBER = [100, 500, 1000, 2000, 3000]
    time_jacobi = []
    time_gaussseidel = []
    time_lu_decomposition = []

    for n in NUMBER:
        matrix_A = create_matrix_A(n)
        vector_b = create_vector_b(n)

        print("N = :", n)
        time_jacobi.append(jacobi(matrix_A, vector_b, n))
        time_gaussseidel.append(gaussseidel(matrix_A, vector_b, n))
        time_lu_decomposition.append(lu_decomposition(matrix_A, vector_b, n))
        print()

    pyplot.plot(NUMBER, time_jacobi, label="Jacobi's method", color="red")
    pyplot.plot(NUMBER, time_gaussseidel, label="Gauss-Seidel's method", color="blue")
    pyplot.plot(NUMBER, time_lu_decomposition, label="LU method", color="green")
    pyplot.legend()
    pyplot.ylabel('Czas [s]')
    pyplot.xlabel('N')
    pyplot.title(' Wykres zależności czasu trwania poszczególnych algorytmów od liczby niewiadomych')
    pyplot.show()





