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

def condMC(A,p):
    return normaMatMC(A,p,p,1000)[0]*normaMatMC(np.linalg.inv(A),p, p,1000)[0]
          
def condExacta(A, p):
    """
    Que devuelva el numero de condicion de A a partir de la formula de
    la ecuacion (1) usando la norma p.
    """
    normaPdeA = normaExacta(A, p)
    normaPdeAinv = normaExacta(np.linalg.inv(A), p)
    nroDeCondicionDeA = normaPdeA * normaPdeAinv
    return nroDeCondicionDeA        
