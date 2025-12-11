import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Paramètres Physiques ---
c = 1540.0        # Vitesse du son (m/s)
f0 = 2.5e6        # Fréquence (2.5 MHz)

# Coordonnées des cibles
x_target1 = -4e-3
z_target1 = -30e-3

x_target2 = 4e-3   # Nouvelle cible à +4mm
z_target2 = -30e-3 # Même profondeur

# Position du capteur
x_sensor = 0.0
z_sensor = 0.0

# Grille de simulation
Nx, Nz = 300, 400
x_range = np.linspace(-15e-3, 15e-3, Nx)
z_range = np.linspace(5e-3, -40e-3, Nz)
X, Z = np.meshgrid(x_range, z_range)

# Définition de l'impulsion
def pulse(t):
    sigma = 0.3 / f0
    return np.cos(2 * np.pi * f0 * t) * np.exp(-(t**2) / (2 * sigma**2))

# --- 2. Configuration de l'Animation ---
fig, ax = plt.subplots(figsize=(8, 6))

# Calcul du temps max nécessaire
dist_Rz = np.abs(z_target1 - 0) # Identique pour les deux
dist_Rrx1 = np.sqrt((x_target1 - x_sensor)**2 + (z_target1 - z_sensor)**2)
dist_Rrx2 = np.sqrt((x_target2 - x_sensor)**2 + (z_target2 - z_sensor)**2)

t_aller = dist_Rz / c
t_retour = max(dist_Rrx1, dist_Rrx2) / c
t_total = t_aller + t_retour + 10e-6 
frames = 100
dt = t_total / frames

# Initialisation image
img = ax.imshow(np.zeros_like(X), extent=[x_range[0]*1e3, x_range[-1]*1e3, z_range[-1]*1e3, z_range[0]*1e3], 
                cmap='RdBu', vmin=-1, vmax=1, aspect='equal', alpha=0.8)

# Éléments fixes
ax.scatter([x_sensor*1e3], [z_sensor*1e3], c='green', marker='s', s=100, label='Capteur (Rx)', zorder=10)
ax.scatter([x_target1*1e3], [z_target1*1e3], c='yellow', marker='o', s=100, label='Cible 1', zorder=10)
ax.scatter([x_target2*1e3], [z_target2*1e3], c='orange', marker='o', s=100, label='Cible 2', zorder=10)

ax.set_xlabel("X [mm]")
ax.set_ylabel("Z [mm]")
ax.set_title("Animation Onde Plane : 2 Cibles")
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='upper right')

text_info = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.7))

def update(frame):
    t = frame * dt
    
    # --- A. Onde Plane (Aller) ---
    tau_tx = np.abs(Z) / c 
    mask_incident = (Z > z_target1) # On arrête l'onde après les cibles
    wave_inc = pulse(t - tau_tx) * mask_incident * 0.8

    # --- B. Onde Sphérique Cible 1 ---
    R1 = np.sqrt((X - x_target1)**2 + (Z - z_target1)**2)
    tau_echo1 = t_aller + (R1 / c)
    wave_refl1 = pulse(t - tau_echo1) * 0.5

    # --- C. Onde Sphérique Cible 2 ---
    R2 = np.sqrt((X - x_target2)**2 + (Z - z_target2)**2)
    tau_echo2 = t_aller + (R2 / c)
    wave_refl2 = pulse(t - tau_echo2) * 0.5
    
    # Somme des trois ondes (Principe de superposition)
    field = wave_inc + wave_refl1 + wave_refl2
    
    img.set_data(field)
    text_info.set_text(f'Temps: {t*1e6:.1f} µs')

    # --- D. Tracé des vecteurs à la fin ---
    if frame > frames - 10:
        # Vecteurs Cible 1
        ax.plot([x_target1*1e3, x_target1*1e3], [0, z_target1*1e3], 'b--', lw=1, alpha=0.5) 
        ax.plot([x_target1*1e3, x_sensor*1e3], [z_target1*1e3, z_sensor*1e3], 'r--', lw=2)
        
        # Vecteurs Cible 2
        ax.plot([x_target2*1e3, x_target2*1e3], [0, z_target2*1e3], 'b--', lw=1, alpha=0.5)
        ax.plot([x_target2*1e3, x_sensor*1e3], [z_target2*1e3, z_sensor*1e3], 'orange', linestyle='--', lw=2)
        
    return img, text_info

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
# ... (tout le code précédent)

# Sauvegarder en GIF
# writer='pillow' est le moteur de création de GIF intégré
# fps=20 définit la fluidité (20 images par seconde)
ani.save('animation_onde.gif', writer='pillow', fps=20)

print("Sauvegarde terminée : animation_onde_deux_cible.gif")

plt.show()

