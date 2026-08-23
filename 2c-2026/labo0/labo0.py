
import numpy as np
# ej 1
def es_cuadrada(A):
    f, c= A.shape
    if f==c:
        return True
    return False
# ej 2
def matriz_identidad(f,c):
    res=np.zeros((f,c))
    for i in range(f):
        res[i,i]=1
    return res
""""
def matrizLU(A):
    n,m = A.shape
    if n!=m:
        return None
    L=matriz_identidad(n,m)
    for i in range(n):
        for j in range(m):
            if A[i,i]!=0:
                pivote= """
    
# ej 3

# ej 4

# ej 5
def traza(A):
    n, m =A.shape
    res=0
    if n!=m:
        return None
    for i in range(n):
        res+A[i,i]
    return res

def test_traza_identidad():
    A = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    assert traza(A) == 3


def test_traza_matriz():
    A = np.array([
        [2, 4, 1],
        [0, 3, 5],
        [7, 8, -1]
    ])
    assert traza(A) == 4
    
# ej 6
def transpuesta(A):
    n, m = A.shape
    res= np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            res[j,i]= A[i,j]
    return res
  # test  
def test_transpuesta_rectangular():
    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    esperada = np.array([
        [1, 4],
        [2, 5],
        [3, 6]
    ])

    assert np.array_equal(transpuesta(A), esperada)
    
# ej 7
def esSimetrica(A):
    n, m = A.shape
    if n!=m:
        return False
    for i in range(n):
        for j in range(m):
           if  A[i,j]!=A[j,i]:
               return False
    return True

def test_esSimetrica_si():
    A = np.array([
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6]
    ])

    assert esSimetrica(A) == True
    
A = np.array([[1, 2], [3, 4]])
print(traza(A))

test_esSimetrica_si()
print("¡Test ejecutado con éxito!")

# ej 8 sirve para cualquier multiplicacion de matrices en general
def calcularAx(A,x):
    n, m=A.shape
    b, c=x.shape
    if m!=b:
        return None
    res=np.zeros((n,c))
    for i in range(n):
        for k in c:
            suma=0
            for j in range(m):
                suma+=A[i,j]*x[j,i]
            res[i][k]=suma
    return res

def test_calcularAx_identidad():
    # Multiplicar por la identidad debe dar la misma matriz
    A = np.array([[1.0, 2.0], 
                  [3.0, 4.0]])
    I = np.identity(2)
    
    resultado = calcularAx(A, I)
    assert np.allclose(resultado, A)

def test_calcularAx_dimensiones_rectangulares():
    # A es 2x3 y x es 3x2 -> el resultado debe ser 2x2
    A = np.array([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0]])
    x = np.array([[7.0, 8.0], 
                  [9.0, 1.0], 
                  [2.0, 3.0]])
    
    esperado = A @ x
    resultado = calcularAx(A, x)
    assert np.allclose(resultado, esperado)

# ej 9
#def intercambiarFilas(A,i,j):

#Ejercicio 10. Desarrollar una función sumar fila multiplo(A, i, j, s) que
#a la fila i le sume la fila j multiplicada por un escalar s. Esta es una operación
# elemental clave en la eliminación gaussiana. La operación debe ser in-place.

def sumar_fila_multiplo(A, i, j, s):
     #minimo la cantidad de filas y cantidad de columnas es mayor a 1
     cant_columnas=A.shape[1]
     
     for k in range (0, cant_columnas):
          A[i,k]=A[i,k]+s*A[j,k]

#Ejercicio 13. Desarrollar una función matrizVandermonde(v), donde v ∈Rn
#y se devuelve la matriz de Vandermonde V ∈Rn×n cuya fila i-ésima corresponde
# con la potencias (i−1)-ésima de los elementos de v.

def matriz_vandermonde (v):
    n=v.size
    A = np.zeros((n,n))
    for i in range (0,n):
     A[i]=v**(i)
    return A
               
    