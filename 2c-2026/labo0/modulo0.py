import numpy as np
#Ejercicio 1
def esCuadrada(A):
    f, c= A.shape
    if f==c:
        return True
    return False

#Ejercicio 4
def diagonal(A):
    m, n = A.shape
    D = np.zeros((m, n))
    #con este if abarcamos los casos en los que #filas<#columnas ó #columnas<#filas, evitando hacer comprobaciones innecesarias o recorrer indices inexistentes
    if m <= n:
        lim = m  
    else:
        lim = n 
    for i in range(lim):
        D[i, i] = A[i, i]
    return D

#Ejercicio 5
def traza(A):
    n,m =A.shape
    res=0
    if n!=m:
        return None
    for i in range(n):
        res+=A[i,i]
    return res

#Ejercicio 6
def traspuesta(A):
    n, m = A.shape
    res= np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            res[j,i]= A[i,j]
    return res

#Ejercicio 7
def esSimetrica(A):
    n, m = A.shape
    if n!=m:
        return False
    for i in range(n):
        for j in range(m):
           if  A[i,j]!=A[j,i]:
               return False
    return True