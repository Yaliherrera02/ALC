import numpy as np
import os
import time 

#EN ESTE ARCHIVO VAN A ESTAR TODAS NUESTRAS FUNCIONES DEL MODULO 

import numpy as np
"""MÓDULO 1"""
# =============================================================================
# Recibe dos numeros x e y, y calcula el error de aproximar 
# x usando y en float64
# =============================================================================
def error(x,y):
    x = np.float64(x)
    y = np.float64(y)
    return abs(y-x)
    
# =============================================================================
# Recibe dos numeros x e y, y calcula el error relativo de aproximar 
# x usando y en float64
# =============================================================================
def error_relativo(x,y):
    return error(x,y) / abs(x)

# =============================================================================
# Devuelve True si ambas matrices son iguales y False en otro caso.
# Considerar que las matrices pueden tener distintas dimensiones,
# ademas de distintos valores.
# =============================================================================

def matricesIguales(A,B):
    tol = 1e-08
    if A.shape != B.shape:
        return False
    col_A=len(A[0])
    filas_A=len(A)
    C = abs(B - A)
    for x in range(0,filas_A,1):
        for y in range(0,col_A,1):
            if C[x][y]>tol:
              return False

    return True

"""MÓDULO 2"""
# =============================================================================
# Recibe un angulo theta y retorna una matriz de 2x2
# que rota un vector dado en un angulo theta
# =============================================================================
"""
Usando coordenadas polares para un v = (x,y) cualquiera, tengo:
x = r.cos(tita0) , y = r.sen(tita0)  *(1)

Luego una rotacion de un angulo tita me daria un w = (x1, y1), entonces:
x1 = r.cos(tita0 + tita) , y1 = r.sen(tita0 + tita)   *(2)

Buscando identidades trigonometricas encuentro que:
cos(tita0 + tita) = cos(tita0).cos(tita) - sen(tita0).sen(tita)  *(3)
sen(tita0 + tita) = sen(tita0).cos(tita) + cos(tita0).sen(tita)  *(4)

Uso *(3) y *(4) y los reemplazo en *(2)
x1 = r.[ cos(tita0).cos(tita) - sen(tita0).sen(tita) ]  *(5)
y1 = r.[ sen(tita0).cos(tita) + cos(tita0).sen(tita) ]  *(6)

Ahora reemplazo cos(tita0) y sen(tita0) de *(5) y *(6) ,usando *(1) 
en las ecuaciones *(5) y *(6)
x1 = r.[ (x/r .cos(tita)) - (y/r .sen(tita)) ]
y1 = r.[ (x/r .cos(tita)) + (y/r .sen(tita)) ]

Me queda finalmente que el vector rotado w = (x1, y1) es igual a:
x1 = x.cos(tita) - y.sen(tita)
y1 = x.cos(tita) + r.sen(tita)"""

# =============================================================================
# Recibe un ángulo theta y una tira de números s,
# y retorna una matriz de 2x2 que rota el vector en un ángulo theta
# y luego lo escala en un factor s
# =============================================================================  
def rota_y_escala(theta,s):
    R =rota(theta)
    E =escala(s)
    res = matriz_x_matriz(E,R)
    return res


# =============================================================================
# Recibe un ángulo theta, una tira de números s (en R2), y un vector b en R2.
# Retorna una matriz de 3x3 que rota el vector en un ángulo theta,
# luego lo escala en un factor s y por último lo mueve en un valor fijo b
# =============================================================================
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
    
# =============================================================================
# Recibe un vector v (en R2), un ángulo theta,
# una tira de números s (en R2), y un vector b en R2.
# Retorna el vector w resultante de aplicar la transformación afín a v
# =============================================================================
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
def rota(theta) :
    lista = [np.cos(theta), -np.sin(theta),
             np.sin(theta), np.cos(theta)]
    A = np.array(lista).reshape(2,2)
    return A

# =============================================================================
# Recibe una tira de números s y retorna una matriz cuadrada de
# n x n, donde n es el tamaño de s.
# La matriz escala la componente i de un vector de Rn en un factor s[i]
# =============================================================================
def escala(s):
    n = len(s)
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = s[i]
    return A       
# A es una matriz diagonal, de esta manera puedo escalar cada componente v[i]
# de un vector v , en un factor s[i]

"""Auxiliares de estos ejercicios"""


def matriz_por_vector(A, v):
    """Recibe una matriz A (n x m) y un vector v (m).
    Retorna un vector w (n) resultado de multiplicar A con v (w = A * v).
    """
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


"""MÓDULO 3"""
def norma(x,p):
    """
    Calcula la norma p del vector x sin np.linalg.norm
    """
    x = np.array(x)
    if p == 1:
        for i in range(len(x)):
            x[i] = abs(x[i])
        return sum(x)
    elif p == 2:
        for i in range(len(x)):
            x[i] = x[i]**2
        return np.sqrt(sum(x))
    elif p == "inf":
        for i in range(len(x)):
            x[i] = abs(x[i])
        return max(x)
    else:
        raise ValueError("p debe ser 1, 2 o np.inf")
    
def normaliza(x,p):
    lista = []
    for i in range(len(x)):
        if norma(x[i],p) != 0:
            lista.append(x[i]/norma(x[i],p))
    return lista


def normaMatMC(A, q, p, Np): #usamos.dot para que corra más rapido los tests, en los módulos siguientes sí usamos nuestra propia implementacion de multiplicación de matrices"""
    """
    Devuelve la norma ||A||_{q,p} y el vector x en el cual se alcanza el maximo.
    Esta versión es para pasar las pruebas de laboratorio.
    """
    # A = [a11,a12,...., a1m] es de tamaño n*m
    #     [a21,a22,...., a2m]
    #     [an1,an2,...., anm]

    vectores_aleatorios = np.random.standard_normal(size=(Np, A.shape[1]))
    # Genera vectores aleatorios en una matriz de np*#columnas de A 
    #   [b11,b12,...., b1m]                       np*m
    #   [b21,b22,...., b2m]
    
    # Normaliza con la norma 'p' 
    vectores_normalizados = normaliza(vectores_aleatorios, p) # va a tener misma dimension que vectores_aleatorios
    
    max_norma = 0
    max_vector = None
    # vectores normalizados:  [c11,c12,...., c1m]  es de np*m
    #                         [c21,c22,...., c2m]
    for x in vectores_normalizados:
        # Aplica la matriz A y calcula la norma de salida 'q'
        norma_actual = norma(np.dot(A, x), q)#vector_resultante= (n*m)@(?,1) donde ?=m 
        if norma_actual > max_norma:         #vector resultante va a ser de n*1
            max_norma = norma_actual
            max_vector = x

    return max_norma, max_vector

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2. (c).
    """
    lista = []
    if p == 1 :
        A = A.T
        for fila in A:
            lista.append(norma(fila,1))
        return max(lista) 
    elif p=="inf" :
        for fila in A:
            lista.append(norma(fila,1))
        return max(lista) 
    else :
        return

def condMC(A, p, Np):
    """
    Devuelve el numero de condicion de A usando la norma inducida p.
    """
    normaPdeA = normaMatMC(A,p,p,Np)[0]
    normaPdeAinv = normaMatMC(np.linalg.inv(A),p,p,Np)[0] #Estamos usando la función inversa de numpy porque es previo a calcular inversa con LU
    nroDeCondiconDeA = normaPdeA * normaPdeAinv
    return nroDeCondiconDeA
    
def condExacta(A, p):
    """
    Que devuelva el numero de condicion de A a partir de la formula de
    la ecuacion (1) usando la norma p.
    """
    normaPdeA = normaExacta(A, p)
    normaPdeAinv = normaExacta(np.linalg.inv(A), p)
    nroDeCondicionDeA = normaPdeA * normaPdeAinv
    return nroDeCondicionDeA

"""MODULO 4"""

def calculaLU(matriz): 
    """   Calcula la factorización LU de la matriz A y retorna las matrices L
   y U, junto con el número de operaciones realizadas. En caso de
  que la matriz no pueda factorizarse retorna None."""
    
    if (matriz is None) or not(esCuadrada(matriz)):
        return None,None,0

    n = matriz.shape[0]
    L =  identidad(n)
    U=matriz.copy()
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

"""Auxiliares de estos ejercicios"""

def esCuadrada(A)->bool:
    col=len(A[0])

    #verifico la cantidad de columnas por fila
    for x in range(1,len(A),1):
        if len(A[x])!=col:
            return False
    return len(A)==col


def identidad(n):
    '''Devuelve matriz identidad'''
    res=np.zeros((n,n))
    for x in range(n):
        res[x][x]=1.0
    return res

def det_diagonal(A):
    '''Determinante de matriz diagonal'''
    res=1
    for x in range(0,len(A),1):
        res=res*A[x][x]
    return res

def vector_canonico(n,i):
    x = [0.0 for _ in range(n)]
    x[i]=1.0
    x=np.array(x)
    return x


def transpuesta(A):  
    n,m = A.shape
    At = np.zeros((m, n))  # matriz de retorno donde ire cambiando valores
    for columna in range(m):
        for fila in range(n):
            At[columna, fila] = A[fila, columna]
    return At

def detMatrizTriangular(U, tol=1e-10):  #funcion auxiliar para ver si U es sing
    determinante = 1 
    for i in range (U.shape[0]):
        determinante = determinante*U[i,i]
    if abs(determinante) < tol : 
        return 0 
    return determinante

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


def esSingular(A):
    try:
        np.linalg.inv(A)
        return False
    except:
        return True



"""MODULO 5"""

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    A = np.asarray(A, "float64")
    p,m = A.shape
    
    if p != m: # Si A no es cuadrada, la hacemos cuadrada agregando filas o columnas de 0s
        if m > p:
            z = np.zeros((m-p,m))
            A = np.concatenate((A,z))
        else:
            z = np.zeros((p,p-m))
            A = np.concatenate((A,z),axis=1)


    n= A.shape[0] # Redefinimos a n y m
    # Creo matriz cuadrada de ceros para Q y R
    Q = np.zeros((n,n))
    R = np.zeros((n,n))
    Qprima = np.zeros((n,n)) # Matriz Q antes de normalizar las columnas
    nops = 0 # Contador de operaciones

    if not norma(A[:,0],2) < tol:

        Q[:,0] = A[:,0] / norma(A[:,0],2) # Primera columna de Q
        R[0,0] = norma(A[:,0],2) # Primer elemento de R
    
    
    for j in range(1,n): # Recorro columnas de A (y Q y R)
        Qprima[:,j] = A[:,j] # Copia cada columna de A a Qprima
        nops += n # Sumo 1 al numero de operaciones por copiar una columna

        for i in range(0,j): # Recorro columnas anteriores de Q
            R[i,j] = sum(Q[:,i] * Qprima[:,j]) # Calculo R[i,j]
            nops += 2*n - 1 # Operaciones del producto escalar
            Qprima[:,j] = Qprima[:,j] - R[i,j] * Q[:,i] # Actualizo Qprima
            nops += 2*n # Operaciones de la resta y multiplicacion por escalar
        
        R[j,j] = norma(Qprima[:,j],2) # Calculo R[j,j]
        nops += 2*n - 1 # Operaciones de la norma
        Q[:,j] = Qprima[:,j] / R[j,j] if R[j,j] > tol else 0 # Normalizo Qprima para obtener Q
        nops += n # Operaciones de la division por escalar
    
    if p > m:
        Q = Q[:,:m] # Reducimos Q
        R = R[:m,:m] # Reducimos R
    
    if p < m:
        Q = Q[:p,:] # Reducimos Q
        R = R[:p,:p] # Reducimos R

    if retorna_nops:
        return Q, R, nops
    
    return Q, R

def QR_con_HH(A, tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    (m, n) = np.shape(A)
    if m < n:
        return None
    R = np.copy(A)  #Inicializo la matriz R como copia de A
    Q = np.eye(m)   #Inicializo Q como matriz identidad de rango m 
    for k in range(n): #para cada columna:
        x = np.copy(R[k:,k]) #toma el vector columna de R de la posición k, comenzando por la fila k. Lo llama x
        x = np.array([x])                                   
        x = transpuesta(x) #traspone x
        a = - np.sign(x[0]) * norma(x,2) #construyo a
        e = np.zeros((m-k,1)) #construyo un vector canonico vertical de tamaño m-k
        e[0][0] = 1                                         
        u = x - (a*e) #construyo u. u genera un proyector que anula los elementos bajo la diagonal de la matriz A en la columna k
        if norma(u,2) > tol: #verifico que la norma 2 de u es mayor a la tolerancia
            u = u/(norma(u,2)) #normalizo u
            H = np.eye(m-k)                                 
            H2 = np.eye(m) #creo una matriz H2 identidad que contendrá a la matriz de householder
            H = H - 2*matriz_x_matriz(u,u.T) # crea la matriz de householder
            H2[k:,k:] = H[:,:] # inserto los valores de la matriz de householder en H2
            R = matriz_x_matriz(H2,R) # multiplica H2 por R para anular los elementos bajo la diagonal de R
            Q = matriz_x_matriz(Q,H2.T) # multiplica a Q por H2. Q será el producto de todas las matrices de householder generadas.

    R = R[:n,:]       #recortamos Q y R para que salga un R cuadrado y un Q con la misma cantidad de columnas que las filas de R
    Q = Q[:,:n]
    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n n elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    metodo = metodo.upper() #MAYUSCULAS
    
    if metodo == 'RH':
        # Retorna Q y R calculadas con Householder
        # La función QR_con_HH no tiene un contador de operaciones
        return QR_con_HH(A, tol)
    
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    
    else:
        # Si el método no es 'RH' ni 'GS'
        return None
    

"""Auxiliares de estos ejercicios"""


def suma_vectorial(x):
    """
    Calcula la suma de todos los elementos del vector x
    """
    x = np.array(x) 
    acumulador = 0.0

    for elemento in x:
        acumulador += elemento
        
    return acumulador


"""MÓDULO 6"""

def metpot2k(A, tol=1e-8,K=1000):
    n = A.shape[0]
    v = np.random.random(n)
    w = fA(2,A,v)
    e = matriz_x_matriz(w.T,v)
    k = 0
    while np.abs(e-1) > tol and k < K:
        v = w
        w = fA(2,A,v)
        e = matriz_x_matriz(w.T,v)
        k += 1
    l = matriz_x_matriz(w.T,matriz_x_matriz(A,w))
    return v, l, k

def diagRH(A, tol=1e-15,K=1000):
    if not(esSimetrica(A)):
        return None
    n = A.shape[0]
    v1,l1, _ = metpot2k(A,tol,K)
    #print(l1)
    #e1 = np.zeros(n)
    e1=vector_canonico(n,0)
    v1=np.asarray(v1, dtype=float).ravel()
    u=np.asarray(e1-v1)


    u_col=u.reshape(-1, 1)
    u_fila=u.reshape(1, -1)

    u_normalizado=norma(u,2)
    #H_v1 = np.eye(n) - 2*(np.outer(u,u)/ (u_normalizado**2))
    H_v1 = np.eye(n) - 2*(matriz_x_matriz(u_col,u_fila)/ (u_normalizado**2))
    #cambiar T por transpuesta manual
    if n == 2:
        S = H_v1
        D = matriz_x_matriz(H_v1,matriz_x_matriz(A,H_v1.T))
    else:
        B = matriz_x_matriz(H_v1,matriz_x_matriz(A,H_v1.T))
        AA = B[1:n,1:n]
        SS, DD = diagRH(AA,tol,K)
        D = np.zeros((n,n))
        D[0][0] = l1.item()
        D[1:n,1:n] = DD
        SSS = np.zeros((n,n))
        SSS[0][0] = 1
        SSS[1:n,1:n] = SS
        S = matriz_x_matriz(H_v1,SSS)
    # Ajusto los elementos que no son diagonales
    for x in range(0,len(D),1):
        for y in range(0,len(D),1):
            if y!=x:
                D[x][y]=0.0

    return S, D
"""Auxiliares de estos ejercicios"""


def vector_canonico(n,i):
    x = [0.0 for _ in range(n)]
    x[i]=1.0
    x=np.array(x)
    return x

def esSimetrica(A,tol=1e-12):
    respuesta = True
    for i in range(len(A)):
        for j in range(len(A)):
            if abs(A[i][j] - A[j][i])>tol:
                print(abs(A[i][j] - A[j][i]))
                respuesta = False
                return False
    return respuesta

def fA(k,A,v):
    w_0 = matriz_x_matriz(A,v)

    w_1 = 0

    if norma(w_0,2) > 0:
        #np.linalg.norm(w_0, ord=2)
        #w_1 = w_0/np.linalg.norm(w_0,ord=2)
        w_normalizado= norma(w_0,2)
        w_1 = w_0/w_normalizado
        for i in range(k-1):
            w_0 = matriz_x_matriz(A,w_1)
            w_normalizado= norma(w_0,2)
            w_1 = w_0/w_normalizado
    return w_1

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
    res=np.array(np.zeros((m,j)))

    for x in range(0,j,1):
        elem=matriz_por_vector(A,B[:,x])
        res[:,x]=elem
    return res


"""MÓDULO 7"""
def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """
    A = np.random.rand(n,n)
    for columna in range(n):
        A[::,columna] = A[::,columna] / norma(A[::,columna],1)
    return A


def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """

    A = np.random.rand(n,n)
    for columna in range(n):
        for fila in range(n):
            if (A[fila,columna] <= thres):
                A[fila,columna] = 1
            else:
                A[fila,columna] = 0
        if norma(A[::,columna],1) != 0 :
            A[::,columna] = A[::,columna] / norma(A[::,columna],1)
        else:
            A[np.random.randint(0,n),columna] = 1
    return A


def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    B = matriz_x_matriz(transpuesta(A),A)
    S , D = diagRH(B)
    n =D.shape[0]
    nuc=[]
    if S is None or D is None:
        return None
    
    for i in range(n):
        autovalor = abs(D[i,i])
        if autovalor <= tol:
            nuc.append(i)
    # sin vectores no nulos en el nucleo
    tam=len(nuc) 
    if tam==0:
        return np.zeros((n,0))
    res= np.zeros((n,tam))
    col =0
    for i in nuc:
        res[:, col] = S[:, i]
        col +=1
    return res

def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    if len(listado) == 0:
        return [{}, (m_filas, n_columnas)]
    
    diccionario = {}
    for i in range(len(listado[0])):
        if abs(listado[2][i]) >= tol:
            diccionario[(listado[0][i], listado[1][i])] = listado[2][i]
    
    return [diccionario, (m_filas, n_columnas)]

def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    matrix_A = np.zeros((A[1][0], A[1][1]))
    for (i, j), value in A[0].items():
        matrix_A[i, j] = value

    w = matriz_por_vector(matrix_A, v)

    return w

"""Auxiliares de estos ejercicios"""

def transpuesta(A): 
    n, m = A.shape
    At = np.zeros((m, n))  # matriz de retorno donde ire cambiando valores
    for columna in range(m):
        for fila in range(n):
            At[columna, fila] = A[fila, columna]
    return At

        
def matriz_por_vector(A, v):
    """Recibe una matriz A (n x m) y un vector v (m).
    Retorna un vector w (n) resultado de multiplicar A con v (w = A * v).
    """
    n, m = A.shape
    w = np.zeros(n) 
    
    for fila in range(n): 
        suma = 0 
        for columna in range(m):
            suma = suma + A[fila, columna] * v[columna]
        w[fila] = suma
        
    return w

"""MODULO 8"""
# los test de svd_reducida segun cuando consultamos nos dijeron que en el calculo de hat_U de numpy no devuelve respectivas matrices reducidas, devuelve el hat_U de la svd completa, por ende los test no coinciden con los calculados de nuestra funcion
# Y tambien observamos que en matrices con grandes nucleos por ejemplo de 12x4 si nos fijabamos directamente en los autovalores de A*A sin sacar raiz cuadrada de estos antes de comparar si son menores a la tolerancia, coincidian con los que devuelve numpy, es por eso que solo hicimos esa modificacion en el if y no en el else de nuestra funcion
def svd_reducida(A,k="max",tol=1e-15):
    """
    A: la matriz de interes (de m x n)
    k: el numero de valores singulares (y vectores) a retener.
    tol: la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) 
    y hatV (matriz de n x k)
    """
    m, n = A.shape
    k_valoresSingulares=[]
    #1) caso donde hay mas filas que columnas, uso la matriz A*A
    if m>=n: # calculo V de A*A
        C = matriz_x_matriz(transpuesta(A),A)
        V, D = diagRH(C)
        f,c =D.shape # dimension de D
        for j in range(f):
            autovalor= D[j,j]
            sigma = abs(autovalor)
            if (sigma>tol):
                k_valoresSingulares.append(np.sqrt(sigma)) # modificacion para que pase el test de 12X4
            else:
                break # para optimizar
        r= len(k_valoresSingulares) # rango de la matriz
        if k=="max" :
            c = r  # Si no nos pasan k por parametro, entonces k = r (svd reducida)
        else:
            c = min(k,r) # si k es menor al rango
        # quito los primeros c columnas
        k_valoresSingulares=k_valoresSingulares[:c]
        #Armo hatU usando los indices
        hatSig = np.array(k_valoresSingulares)
        hatV = V[:, :c] # tomo las primeras c columnas
        hatU= np.zeros((m,c))
        B = multiplicar_matrices(A, hatV)
        for i in range(c):
            hatU[:,i] = B[:,i]/hatSig[i]
            hatU[0:m,i] = hatU[0:m,i]/norma((hatU[0:m,i]),2)
    # caso m<n mas columnas que filas
    # uso AA*
    else:
        C = matriz_x_matriz(A,transpuesta(A))
        U, D = diagRH(C)
        f,c =D.shape # dimension de D
        for j in range(f):
            autovalor= D[j,j]
            sigma = np.sqrt(abs(autovalor))
            if (sigma>tol):
                k_valoresSingulares.append(sigma)
            else:
                break # para optimizar
        r= len(k_valoresSingulares) # rango de la matriz
        if k=="max" :
            c = r  # Si no nos pasan k por parametro, entonces k = r (svd reducida)
        else:
            c = min(k,r) # si k es menor al rango
        # quito los primeros c columnas
        k_valoresSingulares=k_valoresSingulares[:c]
        #Armo hatV usando los indices
        hatSig = np.array(k_valoresSingulares)
        hatU = U[:, :c]
        hatV= np.zeros((n,c))
        B = matriz_x_matriz(transpuesta(A), hatU)
        for i in range(c):
            hatV[:,i] = B[:,i]/hatSig[i]
            hatV[0:n,i] = hatV[0:n,i]/norma((hatV[0:n,i]),2)
    return hatU, hatSig, hatV


#Auxiliares del EJERCICIO 1 del TP

def crearYtGatos(X_gatos): #X_gatos es la matriz embeddings de 1536*n gatos
    """ 
    Crea la matriz de targets Y para los embeddings de gatos
    Parámetros:
        X_gatos (np.array): Matriz de embeddings de gatos (1536 x n_gatos)
    
    Retorna:
        Y_gatos (np.array): Matriz de targets (2 x n_gatos), con la categoría [1,0]^T
    """
    
    # Obtenenemos n (el número de muestras de gatos) a partir de las columnas de X_gatos.
    n_gatos = X_gatos.shape[1]
    
    # Creamos una matriz de ceros de tamaño (2 x n)
    Y_gatos = np.zeros((2, n_gatos))
    
    # Asignamos 1 a la fila 0 (la posición del Gato en el vector [Gato, Perro])
    Y_gatos[0, :] = 1

    return Y_gatos

def crearYtPerros(X_perros):#X_perros es la matriz embeddings de 1536*n perros
    """ 
    Crea la matriz de targets Y para los embeddings de perros
    Parámetros:
        X_perros (np.array): Matriz de embeddings de perros (1536 x n_perros)
        
    Retorna:
        Y_perros (np.array): Matriz de targets (2 x n_perros), con la categoría [0,1]^T
    """

    #Obtenemos n (el número de muestras de perros)
    n_perros = X_perros.shape[1]
    
    # Creamos una matriz de ceros de tamaño (2 x n) 
    Y_perros = np.zeros((2, n_perros))
    
    # Asignar 1 a la fila 1 (la posición del Perro en el vector [Gato, Perro])
    Y_perros[1, :] = 1
    
    return Y_perros

def concatenarHorizontal(A, B):
    """
    Concatena dos matrices de NumPy (A y B) horizontalmente, una al lado de la otra.
    Las matrices deben tener el mismo número de filas.
        
    Retorna:
        C : La matriz resultante de A junto B.
    """
    # Obtenemos las dimensiones de ambas matrices.
    m_A, n = A.shape
    m_B, r = B.shape
    
    # Definimos las dimensiones del nuevo array C (filas de A, columnas de A + columnas de B).
    m = m_A
    dim_C = n + r

    # Creamos el array C con np.zeros (con la dimensión final correcta).
    C = np.zeros((m, dim_C))

    # Copiamos A a las primeras 'n' columnas de C (desde 0 hasta n-1).
    C[:, :n] = A

    # Copiamos B a las columnas restantes de C (desde 'n' hasta 'n+r')
    C[:, n:dim_C] = B


    return C

def multiplicar_matrices(A,B):
    """
    Dada una matriz A y B las multiplica
    """
    n, m = A.shape
    m2, p = B.shape
    if m!=m2:
        return ValueError("las dimensiones de las matrices no coinciden")
    
    AB = np.zeros((n,p))
    for i in range(n):
        for j in range(p): # uso np.sum para optimizar
            AB[i,j] = np.sum(A[i, :]* B[:,j])
    
    return AB

#EJERCICIO 1

def cargarDataset(carpeta):
    """
    Carga los embeddings de gatos y perros desde archivos .npy y los junta en cuatro matrices: 
    X_t, Y_t (entrenamiento) y X_v, Y_v (validación).
    
    Parámetros:
        carpeta (str): Ruta base del directorio que contiene la carpeta 'dataset/cats_and_dogs'.
        
    Retorna:
        X_t, Y_t, X_v, Y_v (tuple of np.array): Los cuatro arrays de embeddings y targets.
    """
    
    # Cargamos archivos de entrenamiento
    cats_train = np.load(os.path.join(carpeta, 'cats_and_dogs/train/cats/efficientnet_b3_embeddings.npy'))
    dogs_train = np.load(os.path.join(carpeta, 'cats_and_dogs/train/dogs/efficientnet_b3_embeddings.npy'))
    
    # Cargamos archivos de validación
    cats_val = np.load(os.path.join(carpeta, 'cats_and_dogs/val/cats/efficientnet_b3_embeddings.npy'))
    dogs_val = np.load(os.path.join(carpeta, 'cats_and_dogs/val/dogs/efficientnet_b3_embeddings.npy'))

    # Entrenamiento (X_t, Y_t)
    X_t = concatenarHorizontal(cats_train, dogs_train)  # Concatenamos los embeddings de gatos y perros en columnas. X_t será de (1536, n_gatos_train + n_perros_train)
    Y_t=concatenarHorizontal((crearYtGatos(cats_train)),crearYtPerros(dogs_train)) #Concatenamos los targest de gatos y perros en columnas. Y_t será de 2*(n_gatos_train + n_perros_train)

    # Validación (X_v, Y_v) ---
    X_v = concatenarHorizontal(cats_val, dogs_val)   # Concatenamos horizontalmente los embeddings de validación. X_v será de (1536, n_gatos_val + n_perros_val)
    Y_v = concatenarHorizontal(crearYtGatos(cats_val), crearYtPerros(dogs_val))  #Creamos los targets para validación y los concatenamos horizontalmente Y_v será de (2, n_gatos_val + n_perros_val)

    return X_t, Y_t, X_v, Y_v


#Auxiliares de EJERCICIO 2 del TP

def calculaCholesky(A): #recibe una matriz simétrica definida positiva
    L,U,c=calculaLU(A)
    n,m=U.shape
    D=np.zeros((n,m))
    for i in range(n):
        d=U[i,i]
        if d>=0:
            diago= np.sqrt(d)
            D[i,i]=diago
    A=multiplicar_matrices(L,D)
    B=multiplicar_matrices(D,transpuesta(L))
    return A 



#EJERCICIO 2

def pinvEcuacionesNormales(X,Y): #X=Xtrain, Y=Ytrain
    """Descripción del problema: Calcular la matriz de pesos W que minimiza el error cuadrático ||Y - WX||_F^2
    usando la Pseudo-Inversa de Moore-Penrose por Ecuaciones Normales y Descomposición de Cholesky.

    Nuestro objetivo es buscar W tal que W*X = Y. Como esto no tiene solución exacta, por Cuadrados Mínimos sabemos que W = YX+.  """
    n, p = X.shape  # n filas, p columnas
    
    # Determinar qué caso usar
    if n > p:
        # Caso (a): X⁺ = (X^T X)^{-1} X^T
        W = caso_a(X, Y)
    elif n < p:
        # Caso (b): X⁺ = X^T (X X^T)^{-1}
        W = caso_b(X, Y)
    else:  # n == p
        # Caso (c): X⁺ = X^{-1}
        W = caso_c(X, Y)
    
    return W

def caso_a(X, Y):

    """"Nuestro objetivo es buscar W tal que W*X = Y. Como esto no tiene solución exacta, por Cuadrados Mínimos sabemos que W = YX+.
        Buscamos X+, donde X+=(X^TX)^-1.X^T. Llamo U=X+, y resolvemos el sistema (X^TX)U=X^T"""
    # 1. Calculamos la Matriz del sistema (Ecuaciones normales)
    # A = X^T X
    Xt = transpuesta(X)
    A = multiplicar_matrices(Xt, X)  # De esta manera obtenemos una matriz cuadrada
    
    #2. Realizamos la descomposición de Cholesky y obtenemos L. A=LL^T donde L es triangular inf y L^T es triangular sup 
    L = calculaCholesky(A)
    Lt=transpuesta(L)
    #Como nosotros teníamos el sistema (X^TX)U=X^T, reemplazando a X^TX por LL^T, ahora simplemente resolvemos el sistema (LL^T)U=X^T, teniendo a U=X+ como incognita
    
    #3. Primer parte: Simplificando nos queda LZ=X^T, una matriz auxiliar que viene de Z=L^T.U
    n_cols_Xt = X.shape[0] # n filas de X=numero de columnas en X^T
    p_filas_Xt = X.shape[1] #P columnas de X= número de filas en X^T

    Z = np.zeros((p_filas_Xt, n_cols_Xt)) #Pues LZ=X^T
    # La idea va a ser ir resolviendo L * z_i = X_t_i para cada columna i del lado derecho (X^T).    

    for i in range(n_cols_Xt):
        Xt_i = Xt[:, i]  # Extraemos i-ésima columna del lado derecho X^T
        
        # Resolvemos L * z_i = X_t_i
        z_i = res_tri(L, Xt_i, inferior=True) #Realizamos la Sustitución hacia adelante, L es inferior)
        Z[:, i] = z_i # Guardamos la columna z_i en la matriz auxiliar Z

    # 4. Segunda Parte: Buscamos resolver (L^T U = Z)  para obtener U, donde U = X+.
    U = np.zeros((p_filas_Xt, n_cols_Xt)) # U es de p x n

    for i in range(n_cols_Xt):
        Z_i = Z[:, i]  # i-ésima columna del lado derecho Z
        
        # Resolvemos L^T * U_i = Z_i 
        U_i = res_tri(Lt, Z_i, inferior=False) #Realizamos Sustitución hacia atrás
        U[:, i] = U_i
        
    pseudoInversaX = U 
    
    W = multiplicar_matrices(Y, pseudoInversaX)
    
    return W

def caso_b (X,Y):
    """"Nuestro objetivo es buscar W tal que W*X = Y. Como esto no tiene solución exacta, por Cuadrados Mínimos sabemos que W = YX+.
        Buscamos X+, donde X+=X^T(XX^T)^-1. Llamo V=X+, y resolvemos el sistema V(XX^T)=X^T"""
    # 1. Calculamos la Matriz del Sistema (Ecuaciones Normales)
    # A = X * X^T
    A = multiplicar_matrices(X,transpuesta(X)) #De esta manera obtenemos una matriz cuadrada de n*n
    
    # 2. Realizamos la descomposición de Cholesky y obtenemos L. A=LL^T donde L es triangular inf y L^T es triangular sup 
    L = calculaCholesky(A)
    #Como nosotros teniamos el sistema V(XX^T)=X^T reemplazando a XX^T por LL^T, ahora simplemente resolvemos el sistema  V(L L^T) = X^T con V=X+, en dos partes.

    #3. Primer Parte: Simplificando nos queda ZL^T=X^T, Z una matriz auxiliar que viene de Z=VL
    #Como tenemos V incognita a izquierda trasponemos en ambos lados y nos queda LZ^T=Xtrain, y resolvemos
    
    Zt_auxiliar = np.zeros(X.shape)
    n_cols = X.shape[1] 
    
    for i in range(n_cols):
        # Extraemos la i-ésima columna de X 
        x_i = X[:, i] 
        
        # Resolvemos el sistema L * z_i^T = x_i
        z_i_transpuesta = res_tri(L, x_i, inferior=True) #Realizamos una sustitución hacia adelante
        
        # Guardamos el resultado como columna i en Zt_auxiliar
        Zt_auxiliar[:, i] = z_i_transpuesta

    #4. Segunda Parte: Buscamos resolver VL=Z.
    # Trasponiendo en ambos lados nos queda L^T.V^T=Z^T  donde V=X+
    V_transpuesta = np.zeros(X.shape) # tendrá la misma dimensión que X: (1536 x 3000).
    Lt=transpuesta(L)
    for i in range(n_cols):
        # Extraemos la i-ésima columna de Z_t  que seria la i-esima fila de Z
        Zt_i = Zt_auxiliar[:, i] 
        #Z_i=Z[i;:]

        # Resolvemos el sistema L^T.V^T = Z^T
        V_i_transpuesta = res_tri(Lt, Zt_i, inferior=False) #Realizamos una sustitución hacia atrás
        
        # Guardamos el resultado como columna i en V^T
        V_transpuesta[:, i] = V_i_transpuesta
    V=transpuesta(V_transpuesta)
    pseudoInversaX=V
    #5. Realizamos la multiplicación W=YX+
    W=multiplicar_matrices(Y,pseudoInversaX)
    return W



def caso_c(X, Y):
    # Caso (c): X⁺ = X⁻¹
    """Nuestro objetivo es buscar W tal que W*X = Y. Como esto no tiene solución exacta, por Cuadrados Mínimos sabemos que W = YX+.
    Buscamos X+, que en nuestro caso X+=X⁻¹, es decir cuando n=p (cuadrada) y de rango completo, la pseudo-inversa es simplemente la inversa.
    Tratamos de despejar W de WX=Y.
    Podemos reutilizar la logica del caso b), pues recordemos: La fórmula del Caso (b) es: X+ = X^T(XX^T)⁻¹.
    
    1. Si X es cuadrada e invertible, aplicamos la propiedad de la inversa de un producto:
       (X^T)⁻¹ = (X^T)⁻¹X⁻¹.
       
    2. Sustituimos en la fórmula: 
       X+ = X^T [ (X^T)⁻¹ X⁻¹ ].
       
    3. Como Xᵀ(Xᵀ)⁻¹ = I, la fórmula se simplifica a: 
       X+ = I X⁻¹ = X⁻¹.
       
    Por lo tanto, al llamar a caso_b(X, Y), estamos resolviendo el sistema 
    V(XX^T) = X^T, y el resultado V es directamente la inversa X⁻¹.
    """
    return caso_b(X, Y)


# Ejercicio 3 pinvSVD
# Para la version de svd_reducida que se utiliza para la funcion pinvSVD es una version optimizada que utiliza la multiplicacion A @ B
# En lugar de la implementacion con bucles para reducir el tiempo de computo, pues la implementacion original es correcta pero extremadamente lenta

def sigmaInversa(S): # entrada vector de valores singulares
    k=len(S)
    sig=np.zeros((k,k)) # la matriz sigma+ tiene la misma dimension pues es cuadrada
    for i in range(k):
        sig[i,i]=1.0/S[i]
    return sig       # salida una matriz cuadrada 
    


def pinvSVD(U, S, V, Y): # recibe U S V de la funcion SVD_reducida_modificada, e Y
    sigmaI= sigmaInversa(S) # como la funcion SVD reducida devuelve un vector de valores singulares, lo convierto a su respectiva matriz inversa
    vxsigmaInversa= multiplicar_matrices(V,sigmaI) # como V es pxn (pues n es el rango de X, entonces tendra n columnas) y sigma es nxn podre multiplicarlos
    Ut= transpuesta(U) # U es nxn (tiene n columnas,pues tiene rango completo), UT tambien es nxn
    Xplus = multiplicar_matrices(vxsigmaInversa,Ut) # vxsigmaInversa es pxn y ut es nxn
    W= multiplicar_matrices(Y,Xplus) # W = Xsigma*Y
    return W
    
# Auxiliares para ejercicio 3 tp    
def svd_reducida_modificada(A,k="max",tol=1e-15):
    """
    A: la matriz de interes (de m x n)
    k: el numero de valores singulares (y vectores) a retener.
    tol: la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) 
    y hatV (matriz de n x k)
    """
    m, n = A.shape
    k_valoresSingulares=[]
    posiciones=[]
    #1) caso donde hay mas filas que columnas, uso la matriz A*A
    if m>=n: # calculo V de A*A
        C = multiplicar_matrices_rapido(transpuesta(A),A)
        V, D = diagRH_modificada(C)
        f,c =D.shape # dimension de D
        for j in range(f):
            autovalor= D[j,j]
            sigma = abs(autovalor)
            if (sigma>tol):
                k_valoresSingulares.append(np.sqrt(sigma)) # modificacion para que pase el test de 12X4
            else:
                break # para optimizar
        r= len(k_valoresSingulares) # rango de la matriz
        if k=="max" :
            c = r  # Si no nos pasan k por parametro, entonces k = r (svd reducida)
        else:
            c = min(k,r) # si k es menor al rango
        # quito los primeros c columnas
        k_valoresSingulares=k_valoresSingulares[:c]
        #Armo hatU usando los indices
        hatSig = np.array(k_valoresSingulares)
        hatV = V[:, :c] # tomo las primeras c columnas
        hatU= np.zeros((m,c))
        B = multiplicar_matrices_rapido(A, hatV)
        for i in range(c):
            hatU[:,i] = B[:,i]/hatSig[i]
            hatU[0:m,i] = hatU[0:m,i]/norma((hatU[0:m,i]),2)
    # caso m<n mas columnas que filas
    # uso AA*
    else:
        C = multiplicar_matrices_rapido(A,transpuesta(A))
        U, D = diagRH(C)
        f,c =D.shape # dimension de D
        for j in range(f):
            autovalor= D[j,j]
            sigma = np.sqrt(abs(autovalor))
            if (sigma>tol):
                k_valoresSingulares.append(sigma)
            else:
                break # para optimizar
        r= len(k_valoresSingulares) # rango de la matriz
        if k=="max" :
            c = r  # Si no nos pasan k por parametro, entonces k = r (svd reducida)
        else:
            c = min(k,r) # si k es menor al rango
        # quito los primeros c columnas
        k_valoresSingulares=k_valoresSingulares[:c]
        posiciones=posiciones[:c]
        #Armo hatV usando los indices
        hatSig = np.array(k_valoresSingulares)
        hatU = U[:, :c]
        hatV= np.zeros((n,c))
        B = multiplicar_matrices_rapido(transpuesta(A), hatU)
        for i in range(c):
            hatV[:,i] = B[:,i]/hatSig[i]
            hatV[0:n,i] = hatV[0:n,i]/norma((hatV[0:n,i]),2)
    return hatU, hatSig, hatV
    
def metpot2k_modificada(A, tol=1e-15, K=1000):
    """
    A: una matriz de n x n
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimación del autovector.
    K: el número máximo de iteraciones a realizarse.
    Retorna: vector v, autovalor lambda y número de iteraciones realizadas k.
    """
    n = A.shape[0]
    v = np.random.rand(n) #vector aleatorio de n elementos
    vv = calcularAx_rapido(A, calcularAx_rapido(A, v)) 
    e = filaxColumna_rapido(vv, v)   # es un prod interno
    k_iter = 0
    while ( abs(e-1) > tol and k_iter < K ):
        v = vv
        if norma(v, 2) > tol :
            v = v / norma(v, 2)   # normalizo
        else :
            v = np.zeros(n)
        vv = calcularAx_rapido(A, calcularAx_rapido(A, v))
        if norma(vv, 2) > tol :
            vv = vv / norma(vv, 2)  # normnalizo
        else :
            vv =  np.zeros(n)      
        e = filaxColumna_rapido(vv, v) # la idea es que en algun momento 
        # esto de 1 o muy cercano a 1. 
        # Esto quiere decir que v y vv son el mismo vector
        # v es el vector de la iteracion anterior y vv el nuevo 
        # si son el mismo o casi identicos, finalizo el ciclo y ese es el avec
        
        k_iter = k_iter+1
    l = filaxColumna_rapido(vv,(calcularAx_rapido(A, vv))) # es el autovalor de este avec
    e = e-1
    return vv, l, k_iter

    

def diagRH_modificada(A, tol=1e-15, K=1000):
    """
    A: una matriz simétrica de n x n
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimación del autovector.
    K: el número máximo de iteraciones a realizarse.
    Retorna: matriz de autovectores S y matriz de autovalores D, tal que A = S D S.T.
    Si la matriz A no es simétrica, debe retornar None.
    """
    n = A.shape[0]
    autovec, autoval, _ = metpot2k_modificada(A, tol, K)
    H_v1 = matriz_Householder_modificada(autovec)
    
    if n == 2:
        S = H_v1
        D = multiplicar_matrices_rapido(H_v1, multiplicar_matrices_rapido(A, transpuesta(H_v1)))

    else:
        B = multiplicar_matrices_rapido(H_v1, multiplicar_matrices_rapido(A, transpuesta(H_v1)))
        A2 = B[1:n,1:n]
        S2, D2 = diagRH(A2, tol, K)
        D = np.zeros((n,n))
        D[0,0] = autoval
        for i in range(1,n):
            D[i,i] = D2[i-1,i-1]
        S3 = np.zeros((n,n)) # S3 es la matriz con 1 en el (0,0) con ceros en el resto de la fila 1 y columna 1 y S2 en S3[1:n,1:n]
        S3[0,0] = 1
        for i in range(1,n):
            for j in range(1,n):
                S3[i,j] = S2[i-1,j-1]
        S = multiplicar_matrices_rapido(H_v1, S3)
        
    for i in range(n):
        S[:,i] /= norma(S[:,i], 2)        
        
    return S, D

def calcularAx_rapido(A, x):
    return A @ x

def filaxColumna_rapido(fila, columna):   # es lo mismo que hacer producto interno
    return float(fila @ columna)



    
def matriz_Householder_modificada(autovec,tol=1e-15):
    """
    Dado un autovector, devuelve la matriz de Householder asociada
    """
    n = autovec.shape[0]
    I = np.eye(n)
    e1 = np.zeros(n)
    e1[0] = 1
    vec_HH = e1 - autovec
    denominador = filaxColumna_rapido(vec_HH, vec_HH) # vec_HH·vec_HH = ||vec_HH||^2
    M = np.zeros((n,n))
    
    # Construcción de la matriz de Householder
    for i in range(n):
        for j in range(n):
            M[i, j] = I[i, j] - 2 * vec_HH[i] * vec_HH[j] / denominador
    return M

def multiplicar_matrices_rapido(A,B):   
    return A @ B

#Auxiliares del EJERCICIO 3 del TP


"""EJERCICIO 4
La función recibe las matrices Q, R de la descomposición QR utilizando HouseHolder, y Y la
matriz de targets de entrenamiento. La función devuelve W"""

def pinvHouseHolder(Q, R, Y): 

    # Primero, el cálculo a hacer el V * R.T = Q

    # Para ello, quiero usar res_tri, ya R es una matriz triangular superior
    # Entonces, necesito transponer ambos lados de la igualdad
    # V*R.T = Q ---> R*V.T = Q.T

    Qtrans = transpuesta(Q)

    # Las dimensiones de Vtrans seran  n x p, donde n son las filas de R y p son las columnas de Qtrans,
    # es decir, las filas de Q
    n = R.shape[0]
    p = Q.shape[0]

    # Ahora bien, como res_tri devuelve arrays en filas, es como si estuviese transponiendo
    # las columnas de V.T, por tanto, me está dando las filas de V. Lo que me permite
    # plantear directamente la matriz V con número de filas p y número de columnas n

    # Lo que haré será armar una matriz de 0s e ir reemplazando sus filas por los arrays
    # devueltos por res_tri en cada iteración del ciclo que tendrá esta función

    V = np.zeros((p,n))
    
    for i in range(p):
        # Extraemos la i-ésima columna de Q.T como vector
        Qt_i = Qtrans[:, i] 
        
        # Resolvemos el sistema R * V.T_i = Q.T_i
        V[i,:] = res_tri(R, Qt_i, inferior=False) # Reemplazamos la fila V_i
        final = time.time()

    # Con V armado, podemos calcular W

    W = matriz_x_matriz(Y,V)
    return W

def pinvGramSchmidt(Q, R, Y): 

    # Primero, el cálculo a hacer el V * R.T = Q

    # Para ello, quiero usar res_tri, ya R es una matriz triangular superior
    # Entonces, necesito transponer ambos lados de la igualdad
    # V*R.T = Q ---> R*V.T = Q.T

    Qtrans = transpuesta(Q)

    # Las dimensiones de Vtrans seran  n x p, donde n son las filas de R y p son las columnas de Qtrans,
    # es decir, las filas de Q
    n = R.shape[0]
    p = Q.shape[0]

    # Ahora bien, como res_tri devuelve arrays en filas, es como si estuviese transponiendo
    # las columnas de V.T, por tanto, me está dando las filas de V. Lo que me permite
    # plantear directamente la matriz V con número de filas p y número de columnas n

    # Lo que haré será armar una matriz de 0s e ir reemplazando sus filas por los arrays
    # devueltos por res_tri en cada iteración del ciclo que tendrá esta función

    V = np.zeros((p,n))
    
    for i in range(p):
        # Extraemos la i-ésima columna de Q.T como vector
        Qt_i = Qtrans[:, i] 
        
        # Resolvemos el sistema R * V.T_i = Q.T_i
        V[i,:] = res_tri(R, Qt_i, inferior=False) # Reemplazamos la fila V_i
  

    # Con V armado, podemos calcular W

    W = matriz_x_matriz(Y,V)
    return W

"""EJERCICIO 5"""

def esPseudoInversa(X, pX, tol = 1e-08):
    
    XpX = matriz_x_matriz(X,pX) # Este es X(X+)
    XpXT = transpuesta(XpX) # Y este es la transpuestas -> Ambas necesarias para la 3ra condición

    XpXX = matriz_x_matriz(XpX,X) # Este es X(X+)X -> Necesaria para la 1ra condición

    pXX = matriz_x_matriz(pX,X) # Este es (X+)X
    pXXT = transpuesta(pXX) # Y este su transpuestas -> Ambas necesasrias para la 4ta condición


    pXXpX = matriz_x_matriz(pXX,pX) # Este es (X+)X(X+) -> Necesaria para la 2da condición

    # Condiciones

    # 1. X(X+)X = X
    # 2. (X+)X(X+) = (X+)
    # 3. (X(X+)).T = X(X+)
    # 4. ((X+)X).T = (X+)X

    # Como tenemos igualdades de matrices A = B, verificamos que A-B = 0, con un grado de tolerancia 1e-8 para cada resta Aij - Bij

    # Primera condición

    if not matricesIgualesConTolerancia(XpXX,X,tol): 
        return False

    # Segunda condición

    if not matricesIgualesConTolerancia(pXXpX,pX,tol): 
        return False
    
    # Tercera Condición

    if not matricesIgualesConTolerancia(XpXT,XpX,tol):
        return False
    
    # Cuarta Condición

    if not matricesIgualesConTolerancia(pXXT,pXX,tol):
        return False
    
    return True

"""Auxiliares de ejercicio 5 del TP"""

def matricesIgualesConTolerancia(A,B,tol= 1e-08):
    #atol=1e-08 -> Tolerencia absoluta de 10**(-8)
    if A.shape != B.shape:
     return False
    col_A=len(A[0])
    filas_A=len(A)
    C = abs(B - A)
    for x in range(0,filas_A,1):
        for y in range(0,col_A,1):
            if C[x][y]>tol:
              return False
    return True

if __name__ == "__main__":
    Xt,Yt,Xv,Yv=cargarDataset('dataset/')
    # Esto va en el init
    Xt2= Xt[:75,1000:1050]
    XtT = Xt2.T
    print("XtT", XtT)
    Q, R = QR_con_GS(XtT)
    # Extendemos Xt.T


    print("XtT", XtT)
    print("Q",Q)
    print("R",R)

    print("Q@R",Q@R)
    print("son iguales:",matricesIguales(XtT,Q@R))


    W_gs = pinvGramSchmidt(Q,R,Yt[:,1000:1050])
    print("W obtenido:\n",W_gs)
    print("W_gs@Xt2",W_gs@Xt2)
    print("diferencia con Yt:", np.linalg.norm(W_gs@Xt2-Yt[:,1000:1050],ord=2))

    print("Dims de W: ", W_gs.shape)
    #---------------------------------------------------------------------------------------

    # Sacamos psinv con la fac QR por HH
    #XtT = Xt.T
    #p, n = XtT.shape
    #Q, R = QR_con_HH(XtT)
    #W_hh = pinvHouseHolder(Q,R,Yt)
    #print("W obtenido:\n",W_hh)
    #print("Dims de W: ", W_hh.shape)
