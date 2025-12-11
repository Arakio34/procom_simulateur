import numpy as np
import matplotlib.pyplot as plt

# 1. La "Vraie Vie" (Signal Continu)
# Imaginons un écho très précis qui arrive à t = 2.5
# On le dessine avec beaucoup de points pour qu'il ait l'air lisse
t_continu = np.linspace(0, 10, 1000)
pulse_continu = np.exp(-((t_continu - 2.5)**2) / 0.5) # Une forme de cloche centrée sur 2.5

# 2. L'Ordinateur (Grille Discrète)
# Ton tableau numpy 'rf' a des cases fixes (ex: t=0, 1, 2, 3...)
# Il ne peut PAS stocker une valeur à t=2.5 ! Il n'a que la case 2 et la case 3.
t_discret = np.arange(0, 11, 1) # [0, 1, 2, 3, 4, 5, ...]

# 3. L'Opération np.interp
# "Quelle est la valeur de la courbe bleue exactement aux endroits des barres grises ?"
valeurs_interpolees = np.interp(t_discret, t_continu, pulse_continu)

# --- Visualisation ---
plt.figure(figsize=(10, 6))

# A. La grille fixe
for t in t_discret:
    plt.axvline(t, color='gray', linestyle='--', alpha=0.5)
plt.text(0.2, 0.9, "Grille de temps fixe (t_discret)", color='gray', transform=plt.gca().transAxes)

# B. Le vrai signal (ce qui se passe physiquement)
plt.plot(t_continu, pulse_continu, label="Vrai Signal (Continu) centré à 2.5", color='blue', alpha=0.4, linewidth=2)

# C. Le résultat de np.interp (ce que l'ordi garde)
plt.plot(t_discret, valeurs_interpolees, 'ro-', label="Résultat np.interp (Discret)", markersize=8)

plt.title("Visualisation de np.interp : Échantillonnage d'un signal décalé")
plt.xlabel("Temps (indices)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()
