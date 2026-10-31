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

"""Ejercicio 4"""

def producto_interno(x, y):
    suma = 0.0
    for i in range(len(x)):
        suma = suma + x[i] * y[i]
    return suma

# ------------------------------------------------------------------
def normalizar_puntos(pts):
    """ traslada el centro de los puntos al origen y los
    escala para que su distancia promedio al origen sea sqrt(diagRH) (diagRH = 2 o 3).
    Devuelve (pts_nuevos, T) con T la matriz (diagRH+1)x(diagRH+1) que hace ese cambio."""
    
    n = pts.shape[0]
    diagRH = pts.shape[1]
    centro = np.zeros(diagRH)
    for i in range(n):
        for j in range(diagRH):
            centro[j] = centro[j] + pts[i, j] / n
    dist_total = 0.0
    for i in range(n):
        s2 = 0.0
        for j in range(diagRH):
            s2 = s2 + (pts[i, j] - centro[j]) ** 2
        dist_total = dist_total + np.sqrt(s2)
    s = np.sqrt(diagRH) / (dist_total / n)
    T = np.eye(diagRH + 1)
    for j in range(diagRH):
        T[j, j] = s
        T[j, diagRH] = -s * centro[j]
    pts_nuevos = np.zeros((n, diagRH))
    for i in range(n):
        for j in range(diagRH):
            pts_nuevos[i, j] = s * (pts[i, j] - centro[j])
    return pts_nuevos, T

def armar_Q(img_pts, obj_pts):
    """Tabla Q del DLT. obj_pts puede ser (n x 3) -> Q de 2n x 12 (ejercicio 2),
    o (n x 2) -> Q de 2n x 9 (ejercicios 3 y 4). Cada punto P = (coordenadas, 1) aporta 2 filas:
        [ P^t   0    -u P^t ]
        [  0   P^t   -v P^t ]"""
    n = obj_pts.shape[0]
    diagRH = obj_pts.shape[1]
    k = diagRH + 1
    Q = np.zeros((2 * n, 3 * k))
    for i in range(n):
        P = np.ones(k)
        for j in range(diagRH):
            P[j] = obj_pts[i, j]
        u = img_pts[i, 0]; v = img_pts[i, 1]
        for j in range(k):
            Q[2 * i, j] = P[j]
            Q[2 * i, 2 * k + j] = -u * P[j]
            Q[2 * i + 1, k + j] = P[j]
            Q[2 * i + 1, 2 * k + j] = -v * P[j]
    return Q

def obtenerQ2d(imgPts, objPts):
    """Ecuación 11 (puntos coplanares): Q de 2n x 9."""
    return armar_Q(imgPts, objPts)

def obtenerQ3d(imgPts, objPts):
    """Ecuación 6 (puntos 3D): Q de 2n x 12."""
    return armar_Q(imgPts, objPts)

def autovector_menor(Q, filas, cols):
    """Autovector de menor autovalor de Q^t Q, acomodado como matriz filas x cols (por filas)."""
    QtQ = matriz_x_matriz(transpuesta(Q), Q)
    S, D = diagRH(QtQ)
    ultima = S.shape[1] - 1                  # diagRH ordena de mayor a menor |autovalor|
    R = np.zeros((filas, cols))
    for i in range(filas):
        for j in range(cols):
            R[i, j] = S[i * cols + j, ultima]
    return R

def resolverDLT3D(Q):
    """Ejercicio 1b / 2b: devuelve M (3x4) tal que minimiza ||Q m||^2 con ||m|| = 1."""
    return autovector_menor(Q, 3, 4)

def resolver_H(Q):
    """Ejercicio 3/4: devuelve H (3x3) = autovector de menor autovalor de Q^t Q."""
    return autovector_menor(Q, 3, 3)

def estimar_homografia(img_pts, obj_pts):
    """Receta completa: normalizar, resolver, deshacer la normalización."""
    img_n, T_img = normalizar_puntos(img_pts)
    obj_n, T_obj = normalizar_puntos(obj_pts)
    H_n = resolver_H(armar_Q(img_n, obj_n))
    H = matriz_x_matriz(inversa(T_img), matriz_x_matriz(H_n, T_obj))
    for i in range(3):                        # dejar H[2,2] = 1 (solo prolijidad)
        for j in range(3):
            if (i, j) != (2, 2):
                H[i, j] = H[i, j] / H[2, 2]
    H[2, 2] = 1.0
    return H

def aplicar_homografia(H, pts):
    """Lleva cada punto (a,b) a otro lugar: q = H*(a,b,1) y se DIVIDE por la 3ra coordenada.
    Con H va cancha->foto; con inversa(H) va foto->cancha."""
    n = pts.shape[0]
    R = np.zeros((n, 2))
    for i in range(n):
        x = H[0, 0] * pts[i, 0] + H[0, 1] * pts[i, 1] + H[0, 2]
        y = H[1, 0] * pts[i, 0] + H[1, 1] * pts[i, 1] + H[1, 2]
        z = H[2, 0] * pts[i, 0] + H[2, 1] * pts[i, 1] + H[2, 2]
        R[i, 0] = x / z
        R[i, 1] = y / z
    return R

def error_reproyeccion(H, img_pts, obj_pts):
    """Qué tan lejos (en píxeles, en promedio cuadrático) caen los puntos proyectados con H
    respecto de donde vos los marcaste. Si da pocos píxeles, H está bien."""
    pred = aplicar_homografia(H, obj_pts)
    n = obj_pts.shape[0]
    suma = 0.0
    for i in range(n):
        dx = pred[i, 0] - img_pts[i, 0]; dy = pred[i, 1] - img_pts[i, 1]
        suma = suma + dx * dx + dy * dy
    return np.sqrt(suma / n)

def circulo(centro, radio, N=50):
    """Puntos de un círculo en METROS, sin usar cos/sin (no están permitidas).
    Usa que con t entre -1 y 1, el punto ((1-t^2)/(1+t^2), 2t/(1+t^2)) recorre la mitad
    derecha de un círculo de radio 1; la mitad izquierda es la derecha espejada."""
    cant = 2 * N + 1
    R = np.zeros((2 * cant, 2))
    for i in range(cant):
        t = (i - N) / N
        x = (1 - t * t) / (1 + t * t)
        y = 2 * t / (1 + t * t)
        R[i, 0] = centro[0] + radio * x
        R[i, 1] = centro[1] + radio * y
        R[cant + i, 0] = centro[0] - radio * x
        R[cant + i, 1] = centro[1] + radio * y
    return R

def recta_vertical(c, ancho, N=2):
    """Puntos de la recta 'U = c' para V entre 0 y ancho (paralela a la línea de fondo)."""
    R = np.zeros((N + 1, 2))
    for i in range(N + 1):
        R[i, 0] = c
        R[i, 1] = ancho * i / N
    return R


# ------------------------------------------------------------------
# PARTE 5: calibración con puntos 3D (ejercicio 2)
# ------------------------------------------------------------------
def determinante3(A):
    """Determinante 3x3 por la regla de Sarrus/cofactores (a mano)."""
    return (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
            - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
            + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))

def estimar_M3d(img_pts, obj_pts, normalizar=True):
    """DLT 3D completo: (opcional) Hartley, armar Q, resolver, deshacer la normalización."""
    if normalizar:
        img_n, T_img = normalizar_puntos(img_pts)
        obj_n, T_obj = normalizar_puntos(obj_pts)
        M_n = resolverDLT3D(obtenerQ3d(img_n, obj_n))
        return matriz_x_matriz(inversa(T_img), matriz_x_matriz(M_n, T_obj))
    return resolverDLT3D(obtenerQ3d(img_pts, obj_pts))

def proyectar_M(M, pts3d):
    """Proyecta puntos 3D (n x 3) con M (3x4): q = M (U,V,W,1), luego se divide por q[2]."""
    n = pts3d.shape[0]
    R = np.zeros((n, 2))
    for i in range(n):
        q = np.zeros(3)
        for f in range(3):
            q[f] = (M[f, 0] * pts3d[i, 0] + M[f, 1] * pts3d[i, 1]
                    + M[f, 2] * pts3d[i, 2] + M[f, 3])
        R[i, 0] = q[0] / q[2]
        R[i, 1] = q[1] / q[2]
    return R

def error_reproyeccion_3d(M, img_pts, obj_pts):
    """RMS (en píxeles) entre los puntos marcados y los proyectados con M."""
    pred = proyectar_M(M, obj_pts)
    n = obj_pts.shape[0]
    suma = 0.0
    for i in range(n):
        suma = suma + (pred[i, 0] - img_pts[i, 0]) ** 2 + (pred[i, 1] - img_pts[i, 1]) ** 2
    return np.sqrt(suma / n)

def armar_M(K, R, t):
    """M = K [R | t]  (3x4)."""
    KR = matriz_x_matriz(K, R)
    Kt = matriz_por_vector(K, t)
    M = np.zeros((3, 4))
    for i in range(3):
        for j in range(3):
            M[i, j] = KR[i, j]
        M[i, 3] = Kt[i]
    return M

def devolverMatricesCalibracion3d(M, imponer_au_av=False, imponer_skew_cero=False):
    """Descompone M = K [R | t] con R ortonormal (R R^t = I, det R = +1) y K triangular
    superior con K33 = 1 (descomposición RQ por Gram-Schmidt empezando desde la ULTIMA fila).
    Si imponer_au_av: se fuerza alfa_u = alfa_v. Si imponer_skew_cero: se fuerza K12 = 0.
    Cuando hay restricciones, K se corrige y luego R y t se recalculan para ese K.
    Devuelve (R, t, K)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for i in range(3):
        for j in range(3):
            A[i, j] = M[i, j]
        b[i] = M[i, 3]
    # M se conoce salvo escala y signo: elegimos el signo con det(A) > 0 (R con det +1)
    if determinante3(A) < 0:
        for i in range(3):
            for j in range(3):
                A[i, j] = -A[i, j]
            b[i] = -b[i]
    a1 = A[0, :]; a2 = A[1, :]; a3 = A[2, :]
    # ---- Gram-Schmidt desde la última fila: A = Kp R, con Kp triangular superior
    n3 = np.sqrt(producto_interno(a3, a3))
    r3 = a3 / n3
    K23 = producto_interno(a2, r3)
    w2 = a2 - K23 * r3
    K22 = np.sqrt(producto_interno(w2, w2))
    r2 = w2 / K22
    K13 = producto_interno(a1, r3)
    K12 = producto_interno(a1, r2)
    w1 = a1 - K13 * r3 - K12 * r2
    K11 = np.sqrt(producto_interno(w1, w1))
    r1 = w1 / K11
    R = np.zeros((3, 3))
    for j in range(3):
        R[0, j] = r1[j]; R[1, j] = r2[j]; R[2, j] = r3[j]
    Kp = np.zeros((3, 3))
    Kp[0, 0] = K11; Kp[0, 1] = K12; Kp[0, 2] = K13
    Kp[1, 1] = K22; Kp[1, 2] = K23; Kp[2, 2] = n3
    t = matriz_por_vector(inversa(Kp), b)      # b = Kp t  (Kp ya incluye la escala)
    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = Kp[i, j] / n3            # K33 = 1
    # ---- restricciones opcionales
    if imponer_au_av or imponer_skew_cero:
        if imponer_au_av:
            prom = (K[0, 0] + K[1, 1]) / 2.0
            K[0, 0] = prom; K[1, 1] = prom
        if imponer_skew_cero:
            K[0, 1] = 0.0
        Kinv = inversa(K)
        Ah = matriz_x_matriz(Kinv, A)          # ~ s R  (casi ortogonal, por las restricciones)
        bh = matriz_por_vector(Kinv, b)
        # R = factor ortogonal más cercano a Ah:  R = Ah (Ah^t Ah)^(-1/2)
        S, D = diagRH(matriz_x_matriz(transpuesta(Ah), Ah))
        Dm12 = np.zeros((3, 3))
        for i in range(3):
            Dm12[i, i] = 1.0 / np.sqrt(D[i, i])
        inv_raiz = matriz_x_matriz(S, matriz_x_matriz(Dm12, transpuesta(S)))
        R = matriz_x_matriz(Ah, inv_raiz)
        s = 0.0
        RtA = matriz_x_matriz(transpuesta(R), Ah)
        for i in range(3):
            s = s + RtA[i, i] / 3.0            # escala: s = traza(R^t Ah)/3
        t = bh / s
    return R, t, K

