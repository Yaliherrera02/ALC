import numpy as np
import matplotlib.pyplot as plt


def rota(theta):
    a =np.cos(theta)
    b = -np.sin(theta)
    c = np.sin(theta)
    d = np.cos(theta)
    M= np.array([[a,b],[c,d]])
    return M

def graficar_vector(v):
    x = v[0]
    y = v[1]
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot([0, x], [0, y])
    ax.scatter(x, y)
    ax.axis('equal')

    plt.show()

if __name__ == "__main__":

    grados = 90
    theta = grados * np.pi / 180
    A=(rota(theta))
    x=np.array([1,0])
    y=(A@x)
    graficar_vector(x)
    graficar_vector(y)