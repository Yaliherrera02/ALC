import numpy as np
import os
from scipy.linalg import null_space
from todos_modulos import *

def agregar_elemento(v):
    """Incrementa en 1 la dimensión del vector, con 1 en la última posición"""
    w = np.ones(len(v) + 1)
    w[:-1] = v
    return w

def ordenar_factores(M):
    """Multiplica M por una matriz que permuta y cambia de signo"""
    p=np.array([[0.,0.,1],[-1,0.,0.],[0.,-1,0.]])
    M=matriz_x_matriz(p,M)
    return M

def construye_T(dimension):
    "Construye la matriz T para ordenar las ecuaciones de u y v"
    I = np.eye(dimension)
    O = np.zeros((dimension, dimension))
    T = np.vstack([np.hstack([I, O, O]),
        np.hstack([O, O, O]),
        np.hstack([O, I, O]),
        np.hstack([O, O, O]),
        np.hstack([I, O, O]),
        np.hstack([O, O, I])])
    return T


def construye_Ai(M,T):
    """construye las matrices de la forma [[U V W 1 0 0 0 0 -Uu -Vu -Wu -u][0 0 0 0 U V W 1 -Uv -Vv -Wv -v]]"""
    dim = M.shape[1]
    z = M.reshape(3*dim, 1)
    A_i = matriz_x_matriz(T,z).reshape(2, 3*dim)
    return A_i

## EJERCICIO 2 A ##
def obtenerQ3d(datos_2d_imagen, datos_3d):
    """"Recibe un elemento 2d y todo el dataset 3D"""
    T=construye_T(4)
    for j in range(0,len(datos_3d),1):
        coordenadas_3d=datos_3d[j]
        coordenadas_3d=agregar_elemento(coordenadas_3d)
        coordenadas_3d=coordenadas_3d.reshape(1, -1)
        par_uv=datos_2d_imagen[j*2:j*2+2] #elijo el par
        par_uv=agregar_elemento(par_uv)
        par_uv=par_uv.reshape(-1,1)
        base_Aj=matriz_x_matriz(par_uv,coordenadas_3d)
        base_Aj=ordenar_factores(base_Aj)
        Aj=construye_Ai(base_Aj,T)
        if j==0:
            A=Aj
        else:
            A=np.vstack((A, Aj)) 
    return A
# EJERCICIO 2B
def resolverDLT3D(Q):
    #construimos A=QtQ
    Qt= transpuesta(Q)
    A = matriz_x_matriz(Qt,Q)
    # diagonalizamos A
    S,D= diagRH(A)
    # busco el menor autovalor
    indice = 0
    for i in range(1,D.shape[0]):
        if D[i,i] < D[indice,indice]:
            indice = i
    # autovector correspondiente
    m = S[:,indice]
    # normalizamos el autovector
    m = m/norma(m,2)
    # construimos la matriz de 3x4
    M = m.reshape(3,4)
    return M

# EJERCICIO 2c
def devolverMatricesCalibracion3d(M):
    # Divido M (3x4) en A (3x3) y b (3,)
    A = M[:, :3]
    b = M[:, 3]

    # Factorización QR de A inversa
    A_inv = inversa(A)
    Q1, R1 = calculaQR(A_inv)

    # Recuperamos K y R provisionales
    K0 = inversa(R1)
    R0 = transpuesta(Q1)

    # Normalizamos K para que K[2,2] = 1
    c = K0[2, 2]
    K = K0 / c

    # Determinamos el signo del determinante de R0
    det = np.linalg.det(R0)

    if det > 0:
        s = 1
    else:
        s = -1

    # Matriz de rotación con determinante positivo
    R = s * R0

    # Factor de escala de la proyección
    escala = c * s

    # Calculamos la traslación
    t = s * matriz_por_vector(R1, b)

    return R, t, K
## EJERCICIO 3  ##
def obtenerQ2d(datos_2d_imagen, datos_2d_objeto):
    """Construye Q para las correspondencias entre puntos 2D."""
    T = construye_T(3)
    for j in range(0, len(datos_2d_objeto), 1):
        coordenadas_2d = datos_2d_objeto[j]
        coordenadas_2d = agregar_elemento(coordenadas_2d)
        coordenadas_2d = coordenadas_2d.reshape(1, -1)
        par_uv = datos_2d_imagen[j*2:j*2+2]
        par_uv = agregar_elemento(par_uv)
        par_uv = par_uv.reshape(-1, 1)
        base_Aj = matriz_x_matriz(par_uv,coordenadas_2d)
        base_Aj = ordenar_factores(base_Aj)
        Aj = construye_Ai(base_Aj, T)
        if j == 0:
            A = Aj
        else:
            A = np.vstack((A, Aj))
    return A

def construir_objPts():
    """Construye las coordenadas físicas de las 77 esquinas internas del tablero."""
    puntos = np.zeros((77, 2))
    k = 0
    for j in range(7):
        for i in range(11):
            puntos[k, 0] = 30 * i  # Coordenada U
            puntos[k, 1] = 30 * j  # Coordenada V
            k += 1
    return puntos

def resolverDLT(Q):
    A = matriz_x_matriz(transpuesta(Q), Q)
    S, D = diagRH(A)

    indice = 0
    for i in range(1, D.shape[0]):
        if D[i,i] < D[indice,indice]:
            indice = i

    m = S[:, indice]
    return m / norma(m, 2)

def resolverQhb(Q): #estamos duplicando código
    h = resolverDLT(Q)
    H = h.reshape(3, 3)
    
    return H

def construir_v(x, y):
    v=np.array([
        x[0] * y[0],
        x[0] * y[1] + x[1] * y[0],
        x[1] * y[1],
        x[0] * y[2] + x[2] * y[0],
        x[1] * y[2] + x[2] * y[1],
        x[2] * y[2]
    ])
    return v

def armarA(Hlst):
    for j in range(len(Hlst)):
        H = Hlst[j]
        h1 = H[:, 0]
        h2 = H[:, 1]
        fila1 = construir_v(h1, h2)
        fila2 = construir_v(h1, h1) - construir_v(h2, h2)
        Aj = np.vstack((fila1, fila2))
        if j == 0:
            A = Aj
        else:
            A = np.vstack((A, Aj))
    return A

def calcularMatrizK2D(A):
    #El vector singular derecho asociado al menor valor singular es nuestra solución aproximada b    
    b = resolverDLT(A)
    # Construyo B simétrica
    B = np.array([
    [b[0], b[1], b[3]],
    [b[1], b[2], b[4]],
    [b[3], b[4], b[5]]])
    if B[0, 0] < 0:
        B = -B
    if esSDP(B):
        print("B es definida positiva")
    else:
        print("B no es definida positiva")
    L = calculaCholesky(B)
    R = L.T
    K = inversa(R)
    K = K / K[2, 2]
    return K



def producto_vectorial(u, v):
    w = np.zeros(3)
    w[0] = u[1]*v[2] - u[2]*v[1]
    w[1] = u[2]*v[0] - u[0]*v[2]
    w[2] = u[0]*v[1] - u[1]*v[0]
    return w

def obtenerPose2D(Hj, K):
    M = matriz_x_matriz(inversa(K) , Hj)
    a1 = M[:, 0]
    a2 = M[:, 1]
    a3 = M[:, 2]
    lam = 1 / norma(a1,2)  
    r1 = lam * a1
    r2 = lam * a2
    r3 = producto_vectorial(r1, r2) 
    R = np.hstack((r1.reshape(3, 1), r2.reshape(3, 1), r3.reshape(3, 1)))
    t = lam * a3
    return R, t

if __name__ == "__main__":
    detected_corners=' calib/3d/detected_corners.txt'
    p_w_corners=' calib/3d/p_W_corners.txt'
    datos_r2=np.loadtxt(detected_corners)
    datos_r3=np.loadtxt(p_w_corners,delimiter=',')    
    img_1_r2=datos_r2[0] #pruebo con la primer imagen
    Q=obtenerQ3d(img_1_r2,datos_r3)
    print(Q)

