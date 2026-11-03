import matplotlib.pyplot as plt
img = plt.imread('/Users/alison/Documents/facu2026/alc/2c-2026/template-alumnos-20261027/futbol/test-gol-enzo.png')
plt.imshow(img)
plt.show()
"""
alc.py - módulo del TP (Calibración y pose de la cámara). Todo hecho "a manopla" con bucles.
Solo usa de numpy: np.array, np.zeros, np.ones, np.eye, np.sqrt (lista permitida del enunciado).

  PARTE 1  álgebra básica: producto matriz-matriz, matriz-vector, traspuesta, producto interno
  PARTE 2  SVD reducida (Jacobi de un lado): svd_reducida
  PARTE 3  inversa por Gauss-Jordan
  PARTE 4  DLT: normalización de Hartley, tabla Q (2D y 3D), homografía, círculo, rectas
  PARTE 5  calibración con puntos 3D: M, descomposición RQ -> K, R, t
"""
import numpy as np

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

def producto_interno(x, y):
    """x[0]*y[0] + x[1]*y[1] + ...  (mide cuánto 'apuntan juntos' dos vectores)."""
    suma = 0.0
    for i in range(len(x)):
        suma = suma + x[i] * y[i]
    return suma

def normalizar(x):
    """Deja el vector con largo 1 (divide por su largo)."""
    largo = np.sqrt(producto_interno(x, x))
    R = np.zeros(len(x))
    for i in range(len(x)):
        R[i] = x[i] / largo
    return R

def matriz_por_vector(A, v):
    """A (m x n) por v (n): es lo mismo que producto_matriz pero con un vector."""
    m = A.shape[0]; n = A.shape[1]
    R = np.zeros(m)
    for i in range(m):
        suma = 0.0
        for k in range(n):
            suma = suma + A[i, k] * v[k]
        R[i] = suma
    return R

# ------------------------------------------------------------------
# PARTE 2: SVD reducida (sin método de la potencia, sin formar A^t A)
# ------------------------------------------------------------------
def svd_reducida(A, tol=1e-15, max_barridos=100):
    """SVD reducida de A (m x n, con m >= n):  A = U diag(S) V^t.
      U: m x n (columnas ortonormales), S: n valores singulares de MAYOR a MENOR,
      V: n x n (ortogonal; la columna j es el vector singular derecho j).
    Método: Jacobi de un lado (Hestenes). Se rotan de a pares las COLUMNAS de A hasta que
    queden ortogonales entre sí. Cuando lo están, el largo de cada columna es un valor
    singular y la columna normalizada es un vector de U. Las mismas rotaciones, acumuladas
    en V, dan los vectores singulares derechos.
    Ventaja: no se forma A^t A, así que no se pierde precisión en los valores singulares chicos."""
    m = A.shape[0]; n = A.shape[1]
    U = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            U[i, j] = A[i, j]
    V = np.eye(n)
    for barrido in range(max_barridos):
        hubo_rotacion = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                # alpha = |col p|^2, beta = |col q|^2, gamma = col p . col q
                alpha = 0.0; beta = 0.0; gamma = 0.0
                for i in range(m):
                    alpha = alpha + U[i, p] * U[i, p]
                    beta = beta + U[i, q] * U[i, q]
                    gamma = gamma + U[i, p] * U[i, q]
                # si las columnas ya son (casi) ortogonales no hay nada que rotar
                if alpha * beta > 0.0 and abs(gamma) > tol * np.sqrt(alpha * beta):
                    hubo_rotacion = True
                    # ángulo de la rotación que anula gamma (c = cos, s = sin, sin usar cos/sin)
                    zeta = (beta - alpha) / (2.0 * gamma)
                    if abs(zeta) > 1e100:             # rotación casi nula: evita overflow de zeta^2
                        t = 1.0 / (2.0 * zeta)
                    elif zeta >= 0:
                        t = 1.0 / (zeta + np.sqrt(1.0 + zeta * zeta))
                    else:
                        t = -1.0 / (-zeta + np.sqrt(1.0 + zeta * zeta))
                    c = 1.0 / np.sqrt(1.0 + t * t)
                    s = c * t
                    # rotar las columnas p y q de U y de V
                    for i in range(m):
                        up = U[i, p]; uq = U[i, q]
                        U[i, p] = c * up - s * uq
                        U[i, q] = s * up + c * uq
                    for i in range(n):
                        vp = V[i, p]; vq = V[i, q]
                        V[i, p] = c * vp - s * vq
                        V[i, q] = s * vp + c * vq
        if not hubo_rotacion:
            break
    # valores singulares = largos de las columnas; normalizar las columnas de U
    S = np.zeros(n)
    for j in range(n):
        suma = 0.0
        for i in range(m):
            suma = suma + U[i, j] * U[i, j]
        S[j] = np.sqrt(suma)
        if S[j] > 1e-300:
            for i in range(m):
                U[i, j] = U[i, j] / S[j]
    # ordenar de mayor a menor (selección), permutando también las columnas de U y V
    for a in range(n - 1):
        mayor = a
        for b in range(a + 1, n):
            if S[b] > S[mayor]:
                mayor = b
        if mayor != a:
            tmp = S[a]; S[a] = S[mayor]; S[mayor] = tmp
            for i in range(m):
                tmp = U[i, a]; U[i, a] = U[i, mayor]; U[i, mayor] = tmp
            for i in range(n):
                tmp = V[i, a]; V[i, a] = V[i, mayor]; V[i, mayor] = tmp
    return U, S, V

# ------------------------------------------------------------------
# PARTE 3: inversa de una matriz (para ir de la foto a la cancha)
# ------------------------------------------------------------------
def inversa(A):
    """Gauss-Jordan: se pega la identidad a la derecha, [A | I], y se hacen operaciones
    de fila hasta que A se convierta en I; lo que queda a la derecha es la inversa."""
    n = A.shape[0]
    M = np.zeros((n, 2 * n))
    for i in range(n):
        for j in range(n):
            M[i, j] = A[i, j]
        M[i, n + i] = 1.0
    for c in range(n):
        # buscar la fila (de c hacia abajo) con el número más grande en la columna c
        mejor = c
        for r in range(c + 1, n):
            if abs(M[r, c]) > abs(M[mejor, c]):
                mejor = r
        # intercambiar filas
        for j in range(2 * n):
            tmp = M[c, j]; M[c, j] = M[mejor, j]; M[mejor, j] = tmp
        # dividir la fila para que el pivote sea 1
        pivote = M[c, c]
        for j in range(2 * n):
            M[c, j] = M[c, j] / pivote
        # anular la columna c en todas las otras filas
        for r in range(n):
            if r != c:
                factor = M[r, c]
                for j in range(2 * n):
                    M[r, j] = M[r, j] - factor * M[c, j]
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i, j] = M[i, n + j]
    return R

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

def obtenerQ2d(imgPts, objPts):
    """Ecuación 11 (puntos coplanares): Q de 2n x 9."""
    return armar_Q(imgPts, objPts)

def obtenerQ3d(imgPts, objPts):
    """Ecuación 6 (puntos 3D): Q de 2n x 12."""
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
    U, S, V = svd_reducida(Q)
    k = S.shape[0] - 1                      # svd_reducida ordena de mayor a menor: el menor es el último
    R = np.zeros((filas, cols))
    for i in range(filas):
        for j in range(cols):
            R[i, j] = V[i * cols + j, k]
    return R

def resolverDLT3D(Q):
    """Ejercicio 1b / 2b: devuelve M (3x4) tal que minimiza ||Q m||^2 con ||m|| = 1."""
    return autovector_menor(Q, 3, 4)

def resolver_H(Q):
    """Ejercicio 3/4: devuelve H (3x3) = vector singular de Q asociado al menor valor singular."""
    return autovector_menor(Q, 3, 3)

def estimar_homografia(img_pts, obj_pts):
    """Receta completa: normalizar, resolver, deshacer la normalización."""
    img_n, T_img = normalizar_puntos(img_pts)
    obj_n, T_obj = normalizar_puntos(obj_pts)
    H_n = resolver_H(armar_Q(img_n, obj_n))
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
        return producto_matriz(inversa(T_img), producto_matriz(M_n, T_obj))
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
    KR = producto_matriz(K, R)
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
        Ah = producto_matriz(Kinv, A)          # ~ s R  (casi ortogonal, por las restricciones)
        bh = matriz_por_vector(Kinv, b)
        # R = factor ortogonal más cercano a Ah. Con Ah = U S V^t, ese factor es R = U V^t
        U_, S_, V_ = svd_reducida(Ah)
        R = producto_matriz(U_, traspuesta(V_))
        s = 0.0
        RtA = producto_matriz(traspuesta(R), Ah)
        for i in range(3):
            s = s + RtA[i, i] / 3.0            # escala: s = traza(R^t Ah)/3
        t = bh / s
    return R, t, K
