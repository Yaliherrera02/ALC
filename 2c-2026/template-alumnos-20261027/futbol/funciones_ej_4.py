"""
alc.py (SOLO lo necesario para el Ejercicio 4 - homografía en la cancha). Todo hecho "a manopla" con bucles.
Solo usa de numpy: np.array, np.zeros, np.ones, np.eye, np.sqrt (lista permitida del enunciado).

  PARTE 1  producto matriz-matriz, traspuesta
  PARTE 2  SVD reducida: svd_reducida (usa norma y diagRH del labo)
  PARTE 3  inversa por Gauss-Jordan
  PARTE 4  DLT 2D: normalización de Hartley, tabla Q, homografía, círculo, recta
"""
import numpy as np
try:
    from todos_modulos import *      # norma, metpot2k, etc. del labo (si están en la carpeta, se usan esas)
except ImportError:
    pass

# --- Respaldo: SOLO si no se importaron norma / metpot2k del labo, se usan estas versiones simples ---
def _norma_respaldo(x, p=2):
    """Norma p de un vector (p = 1, 2 o 'inf'), hecha con bucles."""
    n = x.shape[0]
    if p == 1:
        s = 0.0
        for i in range(n):
            s = s + abs(x[i])
        return s
    if p == 'inf' or p == float('inf'):
        m = 0.0
        for i in range(n):
            if abs(x[i]) > m:
                m = abs(x[i])
        return m
    s = 0.0
    for i in range(n):
        s = s + x[i] * x[i]
    return np.sqrt(s)

def _metpot_respaldo(A, tol=1e-15, K=1000):
    """Método de la potencia: devuelve (autovector unitario, autovalor, cantidad de iteraciones)."""
    n = A.shape[0]
    v = np.zeros(n)
    for i in range(n):
        v[i] = 1.0 + 0.37 * i
    v = v / _norma_respaldo(v, 2)
    it = 0
    for it in range(K):
        w = np.zeros(n)
        for i in range(n):
            suma = 0.0
            for j in range(n):
                suma = suma + A[i, j] * v[j]
            w[i] = suma
        nw = _norma_respaldo(w, 2)
        if nw == 0.0:
            break
        w = w / nw
        dif1 = 0.0; dif2 = 0.0
        for i in range(n):
            dif1 = dif1 + (w[i] - v[i]) ** 2
            dif2 = dif2 + (w[i] + v[i]) ** 2
        v = w
        if np.sqrt(dif1) < tol or np.sqrt(dif2) < tol:
            break
    lam = 0.0
    for i in range(n):
        suma = 0.0
        for j in range(n):
            suma = suma + A[i, j] * v[j]
        lam = lam + v[i] * suma
    return v, np.array([lam]), it + 1

if 'norma' not in globals():
    norma = _norma_respaldo
if 'metpot2k' not in globals():
    metpot2k = _metpot_respaldo

# ------------------------------------------------------------------
# PARTE 1: cuentas básicas hechas a mano
# ------------------------------------------------------------------
def producto_matriz(A, B):
    """Multiplica A (m x n) por B (n x p). Cada casillero (i,j) del resultado
    es 'fila i de A' por 'columna j de B': multiplicar de a pares y sumar."""
    m = A.shape[0]; n = A.shape[1]; p = B.shape[1]
    R = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            suma = 0.0
            for k in range(n):
                suma = suma + A[i, k] * B[k, j]
            R[i, j] = suma
    return R

def traspuesta(A):
    """Da vuelta la matriz: lo que estaba en la fila i, columna j, pasa a la fila j, columna i."""
    m = A.shape[0]; n = A.shape[1]
    R = np.zeros((n, m))
    for i in range(m):
        for j in range(n):
            R[j, i] = A[i, j]
    return R


def _aplanar(x):
    """Pasa un vector (n,), (n,1) o (1,n) -- o un escalar -- a un vector 1D, con bucles."""
    x = np.array(x, dtype=float)
    f = len(x.shape)
    if f == 0:
        R = np.zeros(1)
        R[0] = float(x)
        return R
    cant = 1
    for d in x.shape:
        cant = cant * d
    R = np.zeros(cant)
    if f == 1:
        for i in range(cant):
            R[i] = x[i]
    else:
        k = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                R[k] = x[i, j]
                k = k + 1
    return R

def diagRH(A, tol=1e-15, K=1000):
    """Diagonaliza una matriz simétrica A = S D S^t con reflexiones de Householder:
    método de la potencia para el autovector principal, una reflexión que lo lleva a e1,
    y se repite sobre el bloque que queda (recursión). Devuelve (S, D) o None si A no es simétrica."""
    if not(esSimetrica(A)):
        return None
    n = A.shape[0]
    v1, l1, _ = metpot2k(A, tol, K)
    v1 = _aplanar(v1)
    lam = _aplanar(l1)[0]
    e1 = vector_canonico(n, 0)
    u = np.zeros(n)
    for i in range(n):
        u[i] = e1[i] - v1[i]
    u_col = np.zeros((n, 1))
    u_fila = np.zeros((1, n))
    for i in range(n):
        u_col[i, 0] = u[i]
        u_fila[0, i] = u[i]
    u_normalizado = norma(u, 2)
    if u_normalizado < 1e-14:            # v1 ya es e1: no hace falta reflejar
        H_v1 = np.eye(n)
    else:
        H_v1 = np.eye(n) - 2 * (matriz_x_matriz(u_col, u_fila) / (u_normalizado ** 2))
    if n == 2:
        S = H_v1
        D = matriz_x_matriz(H_v1, matriz_x_matriz(A, transpuesta(H_v1)))
    else:
        B = matriz_x_matriz(H_v1, matriz_x_matriz(A, transpuesta(H_v1)))
        AA = B[1:n, 1:n]
        m = n - 1
        for i in range(m):                # saca la asimetría de redondeo
            for j in range(i + 1, m):
                prom = (AA[i, j] + AA[j, i]) / 2
                AA[i, j] = prom
                AA[j, i] = prom
        SS, DD = diagRH(AA, tol, K)
        D = np.zeros((n, n))
        D[0][0] = lam
        D[1:n, 1:n] = DD
        SSS = np.zeros((n, n))
        SSS[0][0] = 1
        SSS[1:n, 1:n] = SS
        S = matriz_x_matriz(H_v1, SSS)
    # Ajusto los elementos que no son diagonales
    for x in range(0, len(D), 1):
        for y in range(0, len(D), 1):
            if y != x:
                D[x][y] = 0.0
    return S, D

"""Auxiliares de estos ejercicios"""


def vector_canonico(n,i):
    x = [0.0 for _ in range(n)]
    x[i]=1.0
    x=np.array(x)
    return x

import numpy as np

def esSimetrica(A, tol=1e-10):
    n, m = A.shape
    if n != m:
        return False
    
    for i in range(n):
        for j in range(m):
            if abs(A[i,j] - A[j,i]) > tol:
                return False
                
    return True
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# PARTE 2: SVD reducida
# ------------------------------------------------------------------
# svd_reducida usa estos nombres (son los del labo). Se apoyan en las funciones de la PARTE 1:
matriz_x_matriz = producto_matriz
multiplicar_matrices = producto_matriz
transpuesta = traspuesta


# Los test de svd_reducida contra numpy no coinciden porque numpy devuelve U de la SVD completa
# y la nuestra es reducida. En matrices con nucleo grande (ej. 12x4) comparar los autovalores de
# A^t A SIN sacar raiz contra la tolerancia coincidia con numpy; por eso esa modificacion esta solo
# en el if (m >= n) y no en el else.
def svd_reducida(A,k="max",tol=1e-15):
    m, n = A.shape
    k_valoresSingulares=[]
    if m>=n:
        C = matriz_x_matriz(transpuesta(A),A)
        V, D = diagRH(C)
        f,c =D.shape
        for j in range(f):
            autovalor= D[j,j]
            sigma = abs(autovalor)
            if (sigma>tol):
                k_valoresSingulares.append(np.sqrt(sigma))
            else:
                break
        r= len(k_valoresSingulares)
        if k=="max" :
            c = r
        else:
            c = min(k,r)
        k_valoresSingulares=k_valoresSingulares[:c]
        hatSig = np.array(k_valoresSingulares)
        hatV = V[:, :c]
        hatU= np.zeros((m,c))
        B = multiplicar_matrices(A, hatV)
        for i in range(c):
            hatU[:,i] = B[:,i]/hatSig[i]
            hatU[0:m,i] = hatU[0:m,i]/norma((hatU[0:m,i]),2)
    else:
        C = matriz_x_matriz(A,transpuesta(A))
        U, D = diagRH(C)
        f,c =D.shape
        for j in range(f):
            autovalor= D[j,j]
            sigma = np.sqrt(abs(autovalor))
            if (sigma>tol):
                k_valoresSingulares.append(sigma)
            else:
                break
        r= len(k_valoresSingulares)
        if k=="max" :
            c = r
        else:
            c = min(k,r)
        k_valoresSingulares=k_valoresSingulares[:c]
        hatSig = np.array(k_valoresSingulares)
        hatU = U[:, :c]
        hatV= np.zeros((n,c))
        B = matriz_x_matriz(transpuesta(A), hatU)
        for i in range(c):
            hatV[:,i] = B[:,i]/hatSig[i]
            hatV[0:n,i] = hatV[0:n,i]/norma((hatV[0:n,i]),2)
    return hatU, hatSig, hatV


# ------------------------------------------------------------------
# PARTE 3: inversa de una matriz (para ir de la foto a la cancha)
# ------------------------------------------------------------------
def detMatrizTriangular(U, tol=1e-10):  #funcion auxiliar para ver si U es sing
    determinante = 1 
    for i in range (U.shape[0]):
        determinante = determinante*U[i,i]
    if abs(determinante) < tol : 
        return 0 
    return determinante

def esCuadrada(A)->bool:
    col=len(A[0])

    #verifico la cantidad de columnas por fila
    for x in range(1,len(A),1):
        if len(A[x])!=col:
            return False
    return len(A)==col


def calculaLU(matriz): 
    """   Calcula la factorización LU de la matriz A y retorna las matrices L
   y U, junto con el número de operaciones realizadas. En caso de
  que la matriz no pueda factorizarse retorna None."""
    
    if (matriz is None) or not(esCuadrada(matriz)):
        return None,None,0

    n = matriz.shape[0]
    L =  np.eye(n,n)
    U=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            U[i][j]=matriz[i][j]
    cant:int=0
    # Recorre la matriz por columnas.
    for col in range(n-1):
        # Recorre la matriz por filas. Va anulando los elementos debajo de la diagonal
        for fil in range(col+1, n):            
            if U[col][col]==0: #Verifica pivote nulo
                cant=0
                return None,None,cant
            factor = U[fil][col] / U[col][col] #Factor para eliminar el elemento U[fil][col] mediante la fila pivote U[col][col]
            L[fil][col] = factor #Construye matriz L 
            cant+=1
            for k in range(col, n):
                U[fil][k] -= factor * U[col][k] # Actualiza A[fil][k]
            cant+=(n-col)
        cant+=1

    return L, U, cant


def res_tri(L, b, inferior=True):
    """
    Resuelve el sistema Lx = b, donde L es triangular. Se puede indicar
    si es triangular inferior o superior usando el argumento
    'inferior' (por default se asume que es triangular inferior).
    """
    n = L.shape[0]
    y = np.zeros(n)
    if inferior:
        for i in range(n): # Sustitución hacia adelante
            suma= 0.0
            for j in range(i):
                coef = L[i, j]
                sol_ant = y[j]
                suma += coef * sol_ant
            pivote = L[i, i]
            y[i] = (b[i] - suma) / pivote
    else: # Sustitución hacia atrás
        for i in range(n - 1, -1, -1):      
            # El valor a despejar es y[i]. Para eso, a b[i] le restamos los términos q conocemos
            suma = 0.0   
            for j in range(i + 1, n):
                coef = L[i, j]
                sol_post = y[j] # Usamos las soluciones que ya calculamos
                suma += coef * sol_post           
            pivote = L[i, i]
            y[i] = (b[i] - suma) / pivote
    return y

def inversa(A):    
    """
    Para cada elemento de la Identidad resuelve el sistema Ax = Ei. Con Ei vector canónico correspondiente a la posición i. 
     
    """
    #Obtengo dimensión de la matriz original
    n=A.shape[0]
    #creo matriz de 0
    res=np.zeros((n,n))
    I=np.eye((n))
    L, U, co= calculaLU(A) #co =cantidad de operaciones
    #Det A = Det L * Det U
    det_L=detMatrizTriangular(L)
    det_U=detMatrizTriangular(U)

    if (det_L*det_U)==0:
        return None
    for x in range(0,len(I),1):
        # LUx=b
        Up = res_tri(L,I[:,x],True) # Ly=b
        v = res_tri(U,Up,False)     # Ux=y   
        res[:,x]=v
    return res

# ------------------------------------------------------------------
# PARTE 4: homografía (el corazón del ejercicio 4)
# ------------------------------------------------------------------
def normalizar_puntos(pts):
    """Hartley (sirve para puntos 2D o 3D): traslada el centro de los puntos al origen y los
    escala para que su distancia promedio al origen sea sqrt(d) (d = 2 o 3).
    Devuelve (pts_nuevos, T) con T la matriz (d+1)x(d+1) que hace ese cambio."""
    n = pts.shape[0]
    d = pts.shape[1]
    centro = np.zeros(d)
    for i in range(n):
        for j in range(d):
            centro[j] = centro[j] + pts[i, j] / n
    dist_total = 0.0
    for i in range(n):
        s2 = 0.0
        for j in range(d):
            s2 = s2 + (pts[i, j] - centro[j]) ** 2
        dist_total = dist_total + np.sqrt(s2)
    s = np.sqrt(d) / (dist_total / n)
    T = np.eye(d + 1)
    for j in range(d):
        T[j, j] = s
        T[j, d] = -s * centro[j]
    pts_nuevos = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            pts_nuevos[i, j] = s * (pts[i, j] - centro[j])
    return pts_nuevos, T

def armar_Q(img_pts, obj_pts):
    """Tabla Q del DLT. obj_pts puede ser (n x 3) -> Q de 2n x 12 (ejercicio 2),
    o (n x 2) -> Q de 2n x 9 (ejercicios 3 y 4). Cada punto P = (coordenadas, 1) aporta 2 filas:
        [ P^t   0    -u P^t ]
        [  0   P^t   -v P^t ]"""
    n = obj_pts.shape[0]
    d = obj_pts.shape[1]
    k = d + 1
    Q = np.zeros((2 * n, 3 * k))
    for i in range(n):
        P = np.ones(k)
        for j in range(d):
            P[j] = obj_pts[i, j]
        u = img_pts[i, 0]; v = img_pts[i, 1]
        for j in range(k):
            Q[2 * i, j] = P[j]
            Q[2 * i, 2 * k + j] = -u * P[j]
            Q[2 * i + 1, k + j] = P[j]
            Q[2 * i + 1, 2 * k + j] = -v * P[j]
    return Q

#funcion creada antes
def obtenerQ2d(imgPts, objPts):
    """Ecuación 11 (puntos coplanares): Q de 2n x 9."""
    return armar_Q(imgPts, objPts)

def autovector_menor(Q, filas, cols):
    """Resuelve min ||Q h||^2 con ||h|| = 1 usando la SVD reducida de Q:
    h = columna de V asociada al MENOR valor singular, acomodada como matriz filas x cols (por filas)."""
    m = Q.shape[0]; p = Q.shape[1]
    # Con menos filas que columnas (ej. 4 puntos: 8 x 9) la SVD reducida perdería el vector
    # del núcleo. Se completa con filas de ceros (no cambia Q^t Q, ni la solución).
    if m < p:
        Qc = np.zeros((p, p))
        for i in range(m):
            for j in range(p):
                Qc[i, j] = Q[i, j]
        Q = Qc
    # tol = -1: NO se descarta ningun valor singular. Con la tolerancia por defecto (1e-15), si Q tiene
    # un valor singular casi nulo (datos sin ruido, o solo 4 puntos) svd_reducida lo tira como si fuera
    # cero y se perderia justamente el vector del nucleo, que es la solucion que buscamos.
    U, S, V = svd_reducida(Q, "max", -1.0)
    k = S.shape[0] - 1                      # svd_reducida ordena de mayor a menor: el menor es el último
    R = np.zeros((filas, cols))
    for i in range(filas):
        for j in range(cols):
            R[i, j] = V[i * cols + j, k]
    return R

def resolver_H(Q):
    """Ejercicio 3/4: devuelve H (3x3) = vector singular de Q asociado al menor valor singular."""
    return autovector_menor(Q, 3, 3)

def estimar_homografia(img_pts, obj_pts):
    """Receta completa: normalizar, resolver, deshacer la normalización."""
    img_n, T_img = normalizar_puntos(img_pts)
    obj_n, T_obj = normalizar_puntos(obj_pts)
    H_n = resolver_H(obtenerQ2d(img_n, obj_n))
    H = producto_matriz(inversa(T_img), producto_matriz(H_n, T_obj))
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
