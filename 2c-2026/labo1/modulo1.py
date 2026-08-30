"""MÓDULO 1"""
import numpy as np
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
