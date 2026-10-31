import numpy as np
"""
def mult_matrices(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    n, m = A.shape
 
    if B.ndim == 1:
        res = np.zeros(n)
        for i in range(n):
            s = 0.0
            for k in range(m):
                s += A[i, k] * B[k]
            res[i] = s
        return res
    else:
        m2, p = B.shape
        res = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                s = 0.0
                for k in range(m):
                    s += A[i, k] * B[k, j]
                res[i, j] = s
        return res
        """
    
def transpuesta(A):
    n, m = A.shape
    res= np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            res[j,i]= A[i,j]
    return res

    
def multiplicar_matrices(A, B):
    """""
    A es m x p y B es p x n.
    """
    m, p1 = A.shape
    p2, n = B.shape
    if p1 != p2:
        return None
    C = np.zeros((m, n))
    # Bucle i: filas de A (m)
    for i in range(m):
        # Bucle j: columnas de B (n)
        for j in range(n):
            suma = 0
    # Bucle k: Itera sobre los elementos de la fila i de A y la columna j de B (dimensión p)
            for k in range(p1):
                suma += A[i, k] * B[k, j]
            C[i, j] = suma      
    return C

#copiar la otra version!!
# --------------------------------------------------------------------
# calculaLU
# ---------------------------------------------------------------------
def calculaLU(A): # pasa los test locales
    if A is None:
        return None, None, 0
        
    cant_op = 0
    n = A.shape[0]  # filas
    m = A.shape[1]  # columnas
    
    if m != n:
        return None, None, 0
        
    U = A.copy()
    L = np.eye(n, n)
    
    for i in range(n - 1):
        if U[i, i] == 0:
            return None, None, 0
            
        pivote = U[i, i]
        for k in range(i + 1, n):  # filas por debajo del pivote
            factor = U[k, i] / pivote
            cant_op += 1  # 1 división
            L[k, i] = factor  # guardo el factor en L
            
            U[k, i] = 0  # anulamos explícitamente sin gastar operaciones aritméticas
            
            for j in range(i + 1, n):  # <--- Empieza en i + 1
                U[k, j] = U[k, j] - factor * U[i, j]
                cant_op += 2  # 1 multiplicación y 1 resta
                
    return L, U, cant_op

 
# ---------------------------------------------------------------------
# res_tri
# ---------------------------------------------------------------------
def res_tri(A, b, inferior=True):
    """
    Resuelve Ax = b cuando A es triangular (inferior si inferior=True,
    superior si inferior=False), por sustitución hacia adelante/atrás.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
 
    if inferior:
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x[j]
            x[i] = s / A[i, i]
    else:
        for i in range(n - 1, -1, -1):
            s = b[i]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = s / A[i, i]
 
    return x

def detMatrizTriangular(U, tol=1e-10):  #funcion auxiliar para ver si U es sing
    determinante = 1 
    for i in range (U.shape[0]):
        determinante = determinante*U[i,i]
    if abs(determinante) < tol : 
        return 0 
    return determinante

# ---------------------------------------------------------------------
# inversa
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# calculaLDV
# ---------------------------------------------------------------------
def calculaLDV(A):
    """
    Calcula la factorización LDV de la matriz A, de forma tal que A =
    LDV, con L triangular inferior, D diagonal y V triangular
    superior. En caso de que la matriz no pueda factorizarse
    retorna None.
    """

    L,U,cant=calculaLU(A)
    if L is None or U is None:
        return None,None,None
    U_t=(transpuesta(U))
    V_t,D,c=calculaLU(U_t)
    V=np.array(transpuesta(V_t))

    return L,D,V
   
def esSDP(A, atol=1e-16): # cambia la diferencia a 16,por que los profes me dijeron que si se podia 
    """
    Checkea si la matriz A es simétrica definida positiva (SDP) usando
    la factorización LDV.
    """
    res_L,res_D,res_V=calculaLDV(A)
    if res_L is None:
        return False
    L, D, V = res_L,res_D,res_V
    n,m = L.shape 
    f,c = D.shape
    #1)veo si V=L^t
    for i in range(n):
        for j in range(m):
            if abs(A[i,j]-A[j,i]) > atol:
                return False
    #2)veo que todos los elementos de la diagonal sean > 0
    for k in range(f):
        if D[k,k]<=atol:
            return False
    return True

# ---------------------------------------------------------------------
# calculaChoesky
# ---------------------------------------------------------------------

def calculaCholesky(A, atol=1e-10):
    # Chequeo si es o no SDP
    if not esSDP(A):
        return None

    # Descomposición LU 
    L, U, c = calculaLU(A)
    n, m = U.shape
    
    # Construcción de D
    D = np.zeros((n, m))
    for i in range(n):
        d = U[i, i]
        # El pivote debe ser estrictamente positivo
        if d >= atol:
            D[i, i] = np.sqrt(d)
        else:
            return None
        
    # Calculo R tal que A = R @ R.T (R = L @ D)
    R = multiplicar_matrices(L, D)
    return R
