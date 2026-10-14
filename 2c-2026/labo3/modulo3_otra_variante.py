import numpy as np
def norma(x, p):
    """
    Calcula la norma p del vector x sin np.linalg.norm
    """
    x = np.array(x)
    # 1. Norma 1
    if p == 1:
        total = 0
        for i in range(len(x)):
            total += abs(x[i])
        return total

    # 2. Norma infinito
    if p == "inf":
        max_val = 0
        for i in range(len(x)):
            if abs(x[i]) > max_val:
                max_val = abs(x[i])
        return max_val

    # 3. Caso lista o tupla
    if p == [1, "inf"] or p == (1, "inf"):
        return [norma(x, 1), norma(x, "inf")]

    # 4. Cualquier otra norma p (p=2, p=3, etc.)
    total = 0
    for i in range(len(x)):
        total += abs(x[i]) ** p
    return total ** (1 / p)

"""
def norma2(x, p):
   
   #Calcula la norma p del vector x sin np.linalg.norm
    
    x = np.array(x)
    
    # 1. Norma 1
    if p == 1:
        return np.sum(np.abs(x))

    # 2. Norma infinito
    if p == "inf":
        max_val = 0
        for i in range(len(x)):
            if abs(x[i]) > max_val:
                max_val = abs(x[i])
        return max_val

    # 3. Caso lista o tupla
    if p == [1, "inf"] or p == (1, "inf"):
        return [norma(x, 1), norma(x, "inf")]

    # 4. Cualquier otra norma p general (p=2, p=3, etc.)
    return np.sum(np.abs(x) ** p) ** (1 / p)
"""   
"""
def norma3(x, p):
   
    #Calcula la norma p del vector x sin np.linalg.norm 
    x = np.array(x, dtype=float)

    # 1. Norma 1 (modifica x con el valor absoluto)
    if p == 1:
        for i in range(len(x)):
            x[i] = abs(x[i])
        return sum(x)

    # 2. Norma infinito 
    elif p == "inf":
        for i in range(len(x)):
            x[i] = abs(x[i])
        return max(x)

    # 3. Caso lista o tupla
    elif p == [1, "inf"] or p == (1, "inf"):
        return [norma(x, 1), norma(x, "inf")]

    # 4. Cualquier otra norma p general 
    else:
        for i in range(len(x)):
            x[i] = abs(x[i]) ** p
        return sum(x) ** (1 / p)   
"""

def normaliza(x,p):
    lista = []
    for i in range(len(x)):
        if norma(x[i],p) != 0:
            lista.append(x[i]/norma(x[i],p))
    return lista
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

#si tomo todos los vectores x cuya longitud en norma p es exactamente 1,
#  ¿cuál es la longitud máxima (medida con la norma q) que puede alcanzar 
# el vector transformado Ax?
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
        norma_actual = norma(matriz_por_vector(A, x), q)#vector_resultante= (n*m)@(?,1) donde ?=m 
        if norma_actual > max_norma:         #vector resultante va a ser de n*1
            max_norma = norma_actual
            max_vector = x

    return max_norma, max_vector
"""""
def normaMatMC(A, q, p, Np):
    # Genera vectores aleatorios en una matriz de Np x columnas de A
    vectores_aleatorios = np.random.standard_normal(size=(Np, A.shape[1]))
    
    # Normaliza con la norma 'p'
    vectores_normalizados = normaliza(vectores_aleatorios, p)
    
    # Multiplica A por todos los vectores juntos: cada fila es A @ x
    Y = np.dot(vectores_normalizados, A.T)
    
    # Calcula la norma q de cada resultado (un valor por fila)
    normas = np.array([norma(y, q) for y in Y])
    
    # Encuentra el máximo y su vector correspondiente sin el loop manual
    i_max = np.argmax(normas)
    max_norma = normas[i_max]
    max_vector = vectores_normalizados[i_max]

    return max_norma, max_vector
"""
def transpuesta(A):
    n, m = A.shape
    res= np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            res[j,i]= A[i,j]
    return res

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2. (c).
    """
    lista = []
    if p == 1 :
        A = transpuesta(A)
        for fila in A:
            lista.append(norma(fila,1))
        return max(lista) 
    elif p=="inf" :
        for fila in A:
            lista.append(norma(fila,1))
        return max(lista) 
    else :
        return

def maxNorma(A,p): # calcula el la norma 1 o infinito de una matriz
    n,m=A.shape
    max=0
    if p==1:
        B=transpuesta(A) # calculo max por columnas al transponerlo
        for fila in range(m):
            valor=norma(B[fila],1) # 1 porque quiero la suma de la fila para luego ver el max de las filas
            if valor >= max:
                max=valor
    else: # cuando p es inf
        for fila in range(n): # calculo el max por filas
            valor=norma(A[fila],1) # lo mismo, calculo las suma de la fila
            if valor>=max:
                max=valor
    return max

def normaExacta(A, p=[1, 'inf']):
    if p==1 or p=='inf':
        return maxNorma(A,p)
    elif p==[1, 'inf']:
        lista=[]
        lista.append(maxNorma(A,1))
        lista.append(maxNorma(A,'inf'))
        return lista
    else:
        return None
"""
def condMC(A,p):
    return normaMatMC(A,p,p,1000)[0]*normaMatMC(np.linalg.inv(A),p, p,1000)[0]"""
          
def condExacta(A, p):
    """
    Que devuelva el numero de condicion de A a partir de la formula de
    la ecuacion (1) usando la norma p.
    """
    normaPdeA = normaExacta(A, p)
    normaPdeAinv = normaExacta(np.linalg.inv(A), p)
    nroDeCondicionDeA = normaPdeA * normaPdeAinv
    return nroDeCondicionDeA        


assert(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
assert(norma(np.random.rand(10),2)<=np.sqrt(10))
assert(norma(np.random.rand(10),2)>=0)

# Tests normaliza
for x in normaliza([np.array([1]*k) for k in range(1,11)],2):
    assert(np.allclose(norma(x,2),1))
for x in normaliza([np.array([1]*k) for k in range(2,11)],1):
    print(not np.allclose(norma(x,2),1) )
for x in normaliza([np.random.rand(k) for k in range(1,11)],'inf'):
    assert( np.allclose(norma(x,'inf'),1) )

# Tests normaExacta

assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(normaExacta(np.random.random((10,10)),1)<=10)
assert(normaExacta(np.random.random((4,4)),'inf')<=4)

# Test normaMC

nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 

# Test condMC

A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

# Test condExacta

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,1)
normaA_ = normaExacta(A_,1)
condA = condExacta(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,'inf')
normaA_ = normaExacta(A_,'inf')
condA = condExacta(A,'inf')
assert(np.allclose(normaA*normaA_,condA))