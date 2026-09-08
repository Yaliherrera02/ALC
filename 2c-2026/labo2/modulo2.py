import numpy as np
import matplotlib.pyplot as plt


def rota(theta):
    a =np.cos(theta)
    b = -np.sin(theta)
    c = np.sin(theta)
    d = np.cos(theta)
    M= np.array([[a,b],[c,d]])
    return M

def graficar_vector(v):
    x = v[0]
    y = v[1]
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot([0, x], [0, y])
    ax.scatter(x, y)
    ax.axis('equal')

    plt.show()


def escala(s):
    n = len(s)
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i][i] =s[i]
    return mat


def matriz_por_vector(A, v):
    n, m = A.shape
    w = np.zeros(n) 
    
    for fila in range(n): 
        suma = 0 
        for columna in range(m):
            suma = suma + A[fila, columna] * v[columna]
        w[fila] = suma
        
    return w

def matriz_x_matriz(A,B):
    # Si Dim(A)=1 o Dim(B)=1, lo convierte a vector columna.
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    m, n = A.shape
    i, j = B.shape

    if i!=n:
        return None
    res=np.zeros((m,j))

    for x in range(0,j,1):
        elem=matriz_por_vector(A,B[:,x])
        res[:,x]=elem
    return res

def rota_y_escala(theta,s):
    R =rota(theta)
    E =escala(s)
    res = matriz_x_matriz(E,R)
    return res

def afin(theta,s,b):
    mat = rota_y_escala(theta,s)
    res = np.zeros((3,3))
    for i in range(2):
        for j in range(2):
            res[i][j] = mat[i][j]
    res[0][2] = b[0]
    res[1][2] = b[1]
    res[2][2] = 1
    return res

def trans_afin(v, theta, s, b):
    A = afin(theta, s, b)
    n, m = A.shape
    v3 = np.append(v,1)
    w = np.zeros(n-1) 
    for fila in range(n-1):
        suma = 0
        for columna in range(m):
            suma = suma + A[fila,columna]*v3[columna]
        w[fila] = suma
    return w

if __name__ == "__main__":

    grados = 90
    theta = grados * np.pi / 180
    A=(rota(theta))
    x=np.array([1,0])
    y=(A@x)
    graficar_vector(x)
    graficar_vector(y)