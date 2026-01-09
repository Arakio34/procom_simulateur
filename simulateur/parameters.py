import numpy as np

class Parameters:
    def __init__(self, 
                 c=1540.0,      # Vitesse du son (m/s)
                 f0=5e6,        # Fréquence centrale (Hz)
                 fs=40e6,       # Fréquence d'échantillonnage (Hz)
                 fracBW=0.6,    # Bande passante fractionnaire
                 nCycles=2.5,   # Nombre de cycles du pulse
                 Nelem=80,      # Nombre d'éléments de la sonde
                 pitch=0.15e-3, # Entraxe entre les éléments (m)
                 x_span=20e-3,  # Largeur de l'image (m)
                 z_min=10e-3,   # Profondeur minimale (m)
                 z_max=50e-3,   # Profondeur maximale (m)
                 Nx=256,        # Résolution latérale (pixels)
                 Nz=256,        # Résolution en profondeur (pixels)
                 SNR_dB=15.0,    # Rapport Signal/Bruit (dB)
                 p=985.0  # Masse volumique (reference corp humain) [kg/m³]
                ):
        # Propriétés physiques
        self.c = c
        self.f0 = f0
        self.fs = fs
        self.lam = c / f0  # Longueur d'onde
        self.dt = 1.0 / fs # Pas temporel
        self.p = p
        
        # Propriétés du pulse
        self.fracBW = fracBW
        self.nCycles = nCycles
        
        # Propriétés de la sonde
        self.Nelem = Nelem
        self.pitch = pitch
        self.aperture = (Nelem - 1) * pitch # Ouverture totale
        self.x_el = np.linspace(-self.aperture / 2, self.aperture / 2, Nelem)
        
        # Propriétés de l'image (Grille)
        self.x_span = x_span
        self.z_min = z_min
        self.z_max = z_max
        self.Nx = Nx
        self.Nz = Nz
        self.x_img = np.linspace(-x_span / 2, x_span / 2, Nx)
        self.z_img = np.linspace(z_min, z_max, Nz)
        
        # Simulation
        self.SNR_dB = SNR_dB

    @classmethod
    def from_json(cls, data):
        """Permet de créer l'objet à partir d'un dictionnaire (issu d'un JSON)."""
        # On ne récupère que les clés qui correspondent aux arguments du constructeur
        valid_keys = cls.__init__.__code__.co_varnames
        params_dict = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**params_dict)

    def __repr__(self):
        return f"Parameters(c={self.c}, f0={self.f0}, Nelem={self.Nelem}, SNR={self.SNR_dB})"
