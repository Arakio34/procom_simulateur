import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Paramètres Physiques ---
c = 1540.0        # Vitesse du son (m/s)
f0 = 2.5e6        # Fréquence (2.5 MHz pour mieux voir les cycles)
fs = 100e6        # Fréquence d'échantillonnage simulation

# Coordonnées (en mm convertis en mètres)
x_target = -4e-3
z_target = -30e-3 # La cible est "en bas"
x_sensor = 0.0
z_sensor = 0.0

# Grille de simulation
Nx, Nz = 300, 400
x_range = np.linspace(-15e-3, 15e-3, Nx)
z_range = np.linspace(5e-3, -40e-3, Nz) # De haut (+5mm) en bas (-40mm)
X, Z = np.meshgrid(x_range, z_range)

# Définition de l'impulsion (Pulse Gaussien)
def pulse(t):
    sigma = 0.3 / f0 # Durée de l'impulsion
    return np.cos(2 * np.pi * f0 * t) * np.exp(-(t**2) / (2 * sigma**2))

# --- 2. Configuration de l'Animation ---
fig, ax = plt.subplots(figsize=(8, 6))

# Calculs des distances géométriques pour le tracé final
dist_Rz = np.abs(z_target - 0) # Distance aller (Onde Plane verticale)
dist_Rrx = np.sqrt((x_target - x_sensor)**2 + (z_target - z_sensor)**2) # Distance retour

# Temps total de simulation (Aller + Retour + Marge)
t_aller = dist_Rz / c
t_retour = dist_Rrx / c
t_total = t_aller + t_retour + 10e-6 
frames = 100
dt = t_total / frames

# Initialisation de l'image
img = ax.imshow(np.zeros_like(X), extent=[x_range[0]*1e3, x_range[-1]*1e3, z_range[-1]*1e3, z_range[0]*1e3], 
                cmap='RdBu', vmin=-1, vmax=1, aspect='equal', alpha=0.8)

# Ajout des éléments fixes
ax.scatter([x_sensor*1e3], [z_sensor*1e3], c='green', marker='s', s=100, label='Capteur (Rx)', zorder=10)
ax.scatter([x_target*1e3], [z_target*1e3], c='yellow', marker='o', s=100, label='Cible', zorder=10)
ax.set_xlabel("X [mm]")
ax.set_ylabel("Z [mm]")
ax.set_title("Animation Onde Plane + Écho")
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='upper right')

# Textes et lignes pour la fin
line_rz, = ax.plot([], [], 'b--', linewidth=2, label='Rz (Aller)')
line_rrx, = ax.plot([], [], 'r--', linewidth=2, label='Rrx (Retour)')
text_info = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.7))

def update(frame):
    t = frame * dt
    
    # --- A. Onde Plane (Aller) ---
    # Elle part de z=0 et descend vers les z négatifs
    # Le front d'onde est à la position z = -c*t
    # Distance parcourue depuis z=0 est |z|
    # L'onde existe si t correspond au temps de trajet pour atteindre la profondeur Z
    tau_tx = np.abs(Z) / c 
    # Masque: l'onde plane n'existe plus après avoir dépassé la cible (pour clarifier l'image)
    mask_incident = (Z > z_target) 
    wave_inc = pulse(t - tau_tx) * mask_incident * 0.8

    # --- B. Onde Sphérique (Retour/Huygens) ---
    # Elle part de la cible à t_aller
    # Distance radiale depuis la cible
    R_from_target = np.sqrt((X - x_target)**2 + (Z - z_target)**2)
    
    # Le temps "local" de l'écho est t - (temps aller + temps propagation depuis cible)
    tau_echo = t_aller + (R_from_target / c)
    wave_refl = pulse(t - tau_echo) * 0.5 # Amplitude plus faible
    
    # Somme des ondes
    field = wave_inc + wave_refl
    
    img.set_data(field)
    text_info.set_text(f'Temps: {t*1e6:.1f} µs')

    # --- C. Tracé final des vecteurs (Dernières images) ---
    if frame > frames - 10:
        # Tracer Rz (Aller - Vertical)
        # De (x_target, 0) à (x_target, z_target)
        ax.plot([x_target*1e3, x_target*1e3], [0, z_target*1e3], 'b--', lw=2)
        ax.text(x_target*1e3 + 1, z_target*1e3 / 2, 'Rz', color='blue', fontsize=12, fontweight='bold')
        
        # Tracer Rrx (Retour - Oblique)
        # De (x_target, z_target) à (x_sensor, z_sensor)
        ax.plot([x_target*1e3, x_sensor*1e3], [z_target*1e3, z_sensor*1e3], 'r--', lw=2)
        # Annotation au milieu du segment
        mid_x = (x_target + x_sensor) / 2 * 1e3
        mid_z = (z_target + z_sensor) / 2 * 1e3
        ax.text(mid_x + 1, mid_z, 'Rrx', color='red', fontsize=12, fontweight='bold')
        
    return img, line_rz, line_rrx, text_info

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

# Sauvegarder en GIF
# writer='pillow' est le moteur de création de GIF intégré
# fps=20 définit la fluidité (20 images par seconde)

print("Sauvegarde terminée : animation_onde.gif")

plt.show()
