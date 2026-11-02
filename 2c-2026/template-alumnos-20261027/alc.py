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
    M=p@M
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
    A_i = (T @ z).reshape(2, 3*dim)
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
        base_Aj=par_uv@(coordenadas_3d)
        base_Aj=ordenar_factores(base_Aj)
        Aj=construye_Ai(base_Aj,T)
        if j==0:
            A=Aj
        else:
            A=np.vstack((A, Aj)) 
    return A

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
        base_Aj = par_uv @ coordenadas_2d
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


def resolverQhb(Q): #re utilizar la versión del 2
    U, s, Vt = np.linalg.svd(Q, full_matrices=False)
    h = Vt[-1, :]
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
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    #El vector singular derecho asociado al menor valor singular es nuestra solución aproximada b
    b = Vt[-1, :]
    # Construyo B simétrica
    B = np.array([
    [b[0], b[1], b[3]],
    [b[1], b[2], b[4]],
    [b[3], b[4], b[5]]])
    if B[0, 0] < 0:
        B = -B
    autovalores = np.linalg.eigvalsh(B)
    print(autovalores)
    if np.all(autovalores > 0):
        print("B es definida positiva")
    else:
        print("B no es definida positiva")
    L = np.linalg.cholesky(B)
    R = L.T
    K = np.linalg.inv(R)
    K = K / K[2, 2]
    return K


def obtenerPose2D(Hj, K):
    M = np.linalg.inv(K) @ Hj
    a1 = M[:, 0]
    a2 = M[:, 1]
    a3 = M[:, 2]
    lam = 1 / np.linalg.norm(a1)  
    r1 = lam * a1
    r2 = lam * a2
    r3 = np.cross(r1, r2) #armar producto vectorial en r3
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

