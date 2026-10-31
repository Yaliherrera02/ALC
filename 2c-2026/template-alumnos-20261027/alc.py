import numpy as np
import os
from scipy.linalg import null_space

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

def construye_T():
    "Construye la matriz T(24x12) para obtener las ecuaciones de u y v"
    I = np.eye(4)
    O = np.zeros((4, 4))
    T = np.vstack([np.hstack([I, O, O]),
        np.hstack([O, O, O]),
        np.hstack([O, I, O]),
        np.hstack([O, O, O]),
        np.hstack([I, O, O]),
        np.hstack([O, O, I])])
    return T

def construye_Ai(M,T):
    """construye las matrices de la forma [[U V W 1 0 0 0 0 -Uu -Vu -Wu -u][0 0 0 0 U V W 1 -Uv -Vv -Wv -v]]"""
    z = M.reshape(12, 1)
    A_i = (T @ z).reshape(2, 12)
    return A_i

def obtenerQ3d(datos_2d_imagen, datos_3d):
    """"Recibe un elemento 2d y todo el dataset 3D"""
    T=construye_T()
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


if __name__ == "__main__":
    detected_corners=' calib/3d/detected_corners.txt'
    p_w_corners=' calib/3d/p_W_corners.txt'
    datos_r2=np.loadtxt(detected_corners)
    datos_r3=np.loadtxt(p_w_corners,delimiter=',')    
    img_1_r2=datos_r2[0] #pruebo con la primer imagen
    Q=obtenerQ3d(img_1_r2,datos_r3)
    print(Q)