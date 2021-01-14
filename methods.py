from math import sin

d = 5
c = 3
e = 7
f = 5

def create_matrix( a1, a2, a3, N):
    A = []  # [[],[],[],...,[]]
    for i in range(N):
        new_row = []
        for j in range(N):
            if i == j:
                new_row.append(int(a1))
            elif i == j - 1 or i == j + 1:
                new_row.append(int(a2))
            elif i == j - 2 or i == j + 2:
                new_row.append(int(a3))
            else:
                new_row.append(int(0))

        A.append(new_row)
    return A


def create_matrix_A(N):
    a1 = 5 + e
    a2 = -1
    a3 = -1
    return create_matrix(a1, a2, a3, N)

def create_matrix_C(N):
    a1 = 3
    a2 = -1
    a3 = -1
    return create_matrix(a1, a2, a3, N)


def create_vector_b(N):
    b = []

    for i in range(N):
        b.append(sin(i * (f + 1)))

    return b

def norm(vector):  #norma euklidesowa
    value = 0
    for i in range(len(vector)):
        value += pow(vector[i],2)
    return pow(value, 0.5)


def vector_multiply_matrix(vector, matrix):
    new_vector = []  # [[],[],[],...,[]]
    for i in range(len(matrix)):
        sum = 0
        for j in range(len(matrix)):
            sum += vector[j]*matrix[i][j]
        new_vector.append(sum)

    return new_vector

def ones(N):
    new_vector = []

    for i in range(N):
        new_vector.append(int(1))

    return new_vector

def zeros(N):
    new_vector = []

    for i in range(N):
        new_vector.append(int(0))

    return new_vector


def vector_sub_vector(vector1, vector2):
     new_vector = []

     for i in range(len(vector1)):
         new_vector.append(vector1[i] - vector2[i])

     return new_vector

def identity_matrix(N):
    new_matrix = []

    for i in range(N):
        new_row = []
        for j in range(N):
            if i == j:
                new_row.append(int(1))
            else:
                new_row.append(int(0))
        new_matrix.append(new_row)

    return new_matrix


def matrix_multiply_matrix(matrix1, matrix2, N):
    new_matrix = []  # [[],[],[],...,[]]
    for i in range(N): #wiersz
        new_row = []
        for k in range(N): # kolumna
            sum = 0
            for j in range(N): # w kolumnie
                sum += matrix1[i][j] * matrix2[j][k]
            new_row.append(sum)
        new_matrix.append(new_row)

    return new_matrix

def matrix_sub_matrix(matrix1, matrix2,N):
    new_matrix = []
    for i in range(N):
        new_row = []
        for j in range(N):
            new_row.append(matrix1[i][j] - matrix2[i][j])
        new_matrix.append(new_row)
    return new_matrix

def zero_matrix(N):
    new_matrix = []
    for i in range(N):
        new_row = []
        for j in range(N):
            new_row.append(int(0))
        new_matrix.append(new_row)
    return new_matrix

def copy_matrix(matrix):
    copy = []
    for row in matrix:
        new_row = []
        for elem in row:
            new_row.append(elem)
        copy.append(new_row)
    return copy

def copy_vector(vector):
    copy = []
    for elem in vector:
        copy.append(elem)
    return copy

