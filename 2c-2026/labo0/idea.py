import matplotlib.pyplot as plt
import numpy as np

# Configuración general de la figura
fig, ax = plt.subplots(figsize=(10, 8))
x_vals = np.linspace(-3, 3, 300)

# ==========================================
# 1. SUBESPACIO IMAGEN (El "Piso" W)
# ==========================================
# Elegimos una recta inclinada para ver que no siempre es el eje horizontal:
# W = recta y = 0.5 * x
y_imagen = 0.5 * x_vals
ax.plot(x_vals, y_imagen, color='#1f77b4', linewidth=2.5, label='Imagen W (Subespacio sobre el que proyectamos: y = 0.5x)')

# ==========================================
# 2. VECTOR A PROYECTAR
# ==========================================
v = np.array([0.5, 2.5])
ax.scatter(v[0], v[1], color='black', s=90, zorder=6, label=f'Vector v = ({v[0]}, {v[1]})')
ax.annotate(f'v = ({v[0]}, {v[1]})', xy=(v[0], v[1]), xytext=(v[0] + 0.1, v[1] + 0.15),
            fontweight='bold', fontsize=11)

# ==========================================
# 3. PROYECTOR ORTOGONAL (P_ort)
# ==========================================
# Núcleo ortogonal: debe ser perpendicular a la pendiente 0.5 -> pendiente = -1 / 0.5 = -2
# Nu(P_ort) = recta y = -2 * x
y_nu_ort = -2 * x_vals
ax.plot(x_vals, y_nu_ort, color='#2ca02c', linestyle='--', linewidth=1.5, 
        label='Núcleo Ortogonal: W^⟂ (Dirección a 90°: y = -2x)')

# Vector director de W: w = (1, 0.5) = (2, 1) normalizado
w_dir = np.array([2.0, 1.0])
p_ort = (np.dot(v, w_dir) / np.dot(w_dir, w_dir)) * w_dir  # Proyección escalar sobre w

# Rayo de luz ortogonal (une v con p_ort a 90°)
ax.plot([v[0], p_ort[0]], [v[1], p_ort[1]], color='#2ca02c', linestyle=':', linewidth=2,
        label='Rayo Ortogonal (cae a 90°)')
ax.scatter(p_ort[0], p_ort[1], color='#2ca02c', s=90, zorder=6)
ax.annotate(f'P_ort(v) = ({p_ort[0]:.2f}, {p_ort[1]:.2f})', xy=(p_ort[0], p_ort[1]), 
            xytext=(p_ort[0] - 1.2, p_ort[1] - 0.5),
            color='#2ca02c', fontweight='bold', fontsize=10,
            arrowprops=dict(arrowstyle="->", color='#2ca02c', lw=1.2))

# ==========================================
# 4. PROYECTOR OBLICUO (P_obl)
# ==========================================
# Elegimos un núcleo arbitrario no ortogonal, por ejemplo la vertical: recta x = 0 (eje Y)
# Nu(P_obl) = gen{(0, 1)}
ax.axvline(0, color='#d62728', linestyle='--', linewidth=1.5, 
           label='Núcleo Oblicuo (Dirección arbitraria: eje Y, x = 0)')

# El rayo desde v=(0.5, 2.5) viaja paralelo al eje Y (x se mantiene en 0.5) hasta chocar con W:
# En W, y = 0.5 * x -> y = 0.5 * 0.5 = 0.25
p_obl = np.array([v[0], 0.5 * v[0]])

# Rayo de luz oblicuo (baja verticalmente paralelo al núcleo oblicuo)
ax.plot([v[0], p_obl[0]], [v[1], p_obl[1]], color='#d62728', linestyle=':', linewidth=2,
        label='Rayo Oblicuo (cae paralelo al Núcleo Oblicuo)')
ax.scatter(p_obl[0], p_obl[1], color='#d62728', s=90, zorder=6)
ax.annotate(f'P_obl(v) = ({p_obl[0]:.2f}, {p_obl[1]:.2f})', xy=(p_obl[0], p_obl[1]), 
            xytext=(p_obl[0] + 0.3, p_obl[1] - 0.4),
            color='#d62728', fontweight='bold', fontsize=10,
            arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2))

# ==========================================
# 5. AJUSTES VISUALES
# ==========================================
ax.axhline(0, color='gray', linewidth=0.7)
ax.axvline(0, color='gray', linewidth=0.7)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal', adjustable='box')
ax.set_title('Misma Imagen W con dos Núcleos distintos:\nProyector Ortogonal vs. Proyector Oblicuo', fontsize=12, pad=15)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.show()