import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Paramètres Physiques ---
c = 1540.0          # Vitesse (m/s)
f0 = 5e6            # Fréquence (5 MHz)
lam = c / f0        # Longueur d'onde (~0.3 mm)
d = 4 * lam         # Écartement des sources

# Paramètres de l'impulsion unique
sigma_t = 0.5e-6    # Durée de l'impulsion (assez courte)

# Grille de simulation (Zone de 6mm x 6mm)
Nx, Nz = 300, 300
x_range = np.linspace(-4e-3, 4e-3, Nx)
z_range = np.linspace(0, 6e-3, Nz)
X, Z = np.meshgrid(x_range, z_range)

# Position des émetteurs
x1, z1 = -d / 2, 0
x2, z2 = d / 2, 0

# Pré-calcul des distances
R1 = np.sqrt((X - x1)**2 + (Z - z1)**2)
R2 = np.sqrt((X - x2)**2 + (Z - z2)**2)

# Fonction Impulsion Unique (Gaussienne simple)
def pulse_single(t):
    # On décale le temps pour que l'impulsion ne parte pas à t<0
    t_delay = 3 * sigma_t 
    dt = t - t_delay
    
    # Formule du pulse
    envelope = np.exp(-(dt**2) / (2 * sigma_t**2))
    carrier = np.cos(2 * np.pi * f0 * dt)
    
    # Masque pour couper proprement
    return carrier * envelope * (envelope > 1e-3)

# --- 2. Configuration Graphique ---
fig, ax = plt.subplots(figsize=(7, 7))

# Image
img = ax.imshow(np.zeros_like(X), 
                extent=[x_range[0]*1e3, x_range[-1]*1e3, z_range[-1]*1e3, z_range[0]*1e3],
                cmap='RdBu', vmin=-1.5, vmax=1.5, aspect='equal')

# Décoration
ax.set_xlabel("X [mm]")
ax.set_ylabel("Profondeur Z [mm]")
ax.set_title("Interférence Destructive : Pulse Unique")
ax.scatter([x1*1e3, x2*1e3], [z1*1e3, z2*1e3], c='black', s=50, zorder=10, label='Sources')

# --- Tracé des Lignes d'Annulation ---
for n in range(-5, 5):
    path_diff = np.abs(R2 - R1)
    target_diff = (n + 0.5) * lam
    
    # Masque fin pour les lignes
    mask_nodal = np.abs(path_diff - target_diff) < (lam * 0.03)
    
    if np.any(mask_nodal):
        ax.contour(X*1e3, Z*1e3, mask_nodal, levels=[0.5], colors='lime', linewidths=1, alpha=0.5)

ax.legend(loc='upper right')

# --- 3. Animation ---
def update(frame):
    # Le temps avance
    t = frame * 0.05e-6 
    
    # Calcul des deux pulses uniques
    sig1 = pulse_single(t - R1/c)
    sig2 = pulse_single(t - R2/c)
    
    # Somme
    field = sig1 + sig2
    
    img.set_data(field)
    return img,

# On génère assez de frames pour voir le pulse traverser toute la zone
ani = animation.FuncAnimation(fig, update, frames=120, interval=40, blit=False)

plt.show()

# ani.save('single_pulse_interference.gif', writer='pillow', fps=20)
