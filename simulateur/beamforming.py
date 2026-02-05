import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import hann

def beamforming(params, rf, nelem=None, snr_db=None):
    """
    Beamforming DAS classique (Delay And Sum).
    """
    c = params.c
    f0 = params.f0
    fs = params.fs
    x_span = params.x_span
    z_min = params.z_min
    z_max = params.z_max
    Nx = params.Nx
    Nz = params.Nz
    Nelem = params.Nelem if nelem is None else nelem
    SNR_dB = params.SNR_dB if snr_db is None else snr_db

    x_img    = np.linspace(-x_span / 2, x_span / 2, Nx)
    z_img    = np.linspace(z_min, z_max, Nz)
    
    # Grille de reconstruction
    Xg, Zg = np.meshgrid(x_img, z_img, indexing='xy')  

    bmode_lin = np.zeros((Nz, Nx), dtype=np.float32)
    y_align   = np.zeros((Nelem, Nx, Nz), dtype=np.float32)

    pitch = params.pitch
    aperture = (Nelem - 1) * pitch
    x_el     = np.linspace(-aperture / 2, aperture / 2, Nelem)

    # Axe temporel pour interpolation
    z_max_toa = z_max / c
    r_max     = np.sqrt((x_span / 2 + aperture / 2) ** 2 + z_max ** 2)
    t_max     = z_max_toa + r_max / c + 2 / f0
    t         = np.arange(0.0, t_max + 1.0 / fs, 1.0 / fs)

    # Apodisation (Hann window)
    apo_rx = hann(Nelem)

    eps = np.finfo(np.float32).eps

    # --- Boucle DAS ---
    for ix in range(Nx):
        x0 = x_img[ix]
        zz = z_img
        # On peut vectoriser cette boucle sur Nelem pour aller plus vite, 
        # mais on garde la structure lisible pour l'instant.
        for n in range(Nelem):
            dx = x0 - x_el[n]
            Rrx = np.sqrt(dx ** 2 + zz ** 2)
            # Délai : Onde plane (z/c) + Retour focalisé (Rrx/c)
            # Note: Si simulation focalisée en émission, remplacer zz/c par dist_foc
            tau_tot = (zz / c) + (Rrx / c)
            
            y_n = np.interp(tau_tot, t, rf[:, n], left=0.0, right=0.0)
            y_align[n, ix, :] = y_n.astype(np.float32)

        tmp  = y_align[:, ix, :]     # (Nelem, Nz)
        tmp2 = apo_rx[:, None] * tmp
        y_sum = np.sum(tmp2, axis=0)  # (Nz,)
        bmode_lin[:, ix] = y_sum.astype(np.float32)

    # --- Enveloppe & Log ---
    pad_width = 64
    bmode_padded = np.pad(bmode_lin, ((pad_width, pad_width), (0, 0)), mode='constant')
    analytic_padded = hilbert(bmode_padded, axis=0)
    analytic = analytic_padded[pad_width:-pad_width, :]

    env      = np.abs(analytic)
    env_max  = np.max(env + eps)
    env      = env / env_max # Normalisation [0, 1]
    bmode_dB = 20 * np.log10(env + eps)

    meta = {
        'c': c,
        'f0': f0,
        'fs': fs,
        'Nelem': Nelem,
        'pitch': pitch,
        'x_el': x_el,
        'x_img': x_img,
        'z_img': z_img,
        'SNR_dB': SNR_dB,
    }

    data = {
        'rf': rf,
        't': t,
        'x_el': x_el,
        'x_img': x_img,
        'z_img': z_img,
        'bmode_dB': bmode_dB,
        'env': env, 
        'meta': meta,
    }
    return data


def mvdr_beamforming(params, rf, nelem=None, snr_db=None, regularization=0.1):
    """
    Implémentation MVDR/Capon optimisée pour la génération de dataset.
    """
    c = params.c
    f0 = params.f0
    fs = params.fs
    x_span = params.x_span
    z_min = params.z_min
    z_max = params.z_max
    
    # Résolution augmentée pour l'entraînement (128x128)
    # 64x64 est trop flou pour apprendre des détails fins
    Nx, Nz = 128, 128 
    
    x_img    = np.linspace(-x_span / 2, x_span / 2, Nx)
    z_img    = np.linspace(z_min, z_max, Nz)

    # --- 1. Signal Analytique (Hilbert) ---
    rf_analytic = hilbert(rf, axis=0) 

    Nelem = params.Nelem if nelem is None else nelem
    SNR_dB = params.SNR_dB if snr_db is None else snr_db
    pitch = params.pitch
    aperture = (Nelem - 1) * pitch
    x_el     = np.linspace(-aperture / 2, aperture / 2, Nelem)
    bmode_mvdr = np.zeros((Nz, Nx), dtype=np.float32)
    y_align   = np.zeros((Nelem, Nx, Nz), dtype=np.complex64)

    # Axe temporel
    z_max_toa = z_max / c
    r_max     = np.sqrt((x_span / 2 + aperture / 2) ** 2 + z_max ** 2)
    t_max     = z_max_toa + r_max / c + 2 / f0
    t         = np.arange(0.0, t_max + 1.0 / fs, 1.0 / fs)
    
    print(f"MVDR Grid: {Nx}x{Nz} | Alignement des signaux...")
    
    # --- 2. Alignement (Delay) ---
    for ix in range(Nx):
        x0 = x_img[ix]
        for n in range(Nelem):
            dx = x0 - x_el[n]
            Rrx = np.sqrt(dx ** 2 + z_img ** 2) 
            tau_tot = (z_img / c) + (Rrx / c)
            
            # Interpolation complexe manuelle
            val_real = np.interp(tau_tot, t, np.real(rf_analytic[:, n]), left=0.0, right=0.0)
            val_imag = np.interp(tau_tot, t, np.imag(rf_analytic[:, n]), left=0.0, right=0.0)
            y_align[n, ix, :] = val_real + 1j * val_imag

    print("Calcul des poids (Inv Covariance)...")

    # --- 3. Calcul MVDR ---
    a = np.ones((Nelem, 1), dtype=np.complex64) # Steering vector (ones car déjà aligné)
    eye_N = np.eye(Nelem)

    rf_mvdr = np.zeros((Nz, Nx), dtype=np.float32)
    for iz in range(Nz):
        for ix in range(Nx):
            # Snapshot
            x_vec = y_align[:, ix, iz].reshape(-1, 1)
            # Covariance R = x * x^H
            # Pour stabilité, on peut ajouter du moyennage spatial (sub-array smoothing)
            # Ici on reste sur du Diagonal Loading simple
            R = x_vec @ x_vec.conj().T
            
            power = np.real(np.trace(R))
            if power < 1e-12: continue
                
            # Regularisation (Diagonal Loading)
            delta = (regularization * power / Nelem) 
            R_loaded = R + delta * eye_N
            
            # Inversion (solve est plus stable que inv)
            try:
                # w = (R^-1 * a) / (a^H * R^-1 * a)
                # On calcule num = R^-1 * a en résolvant R * num = a
                num = np.linalg.solve(R_loaded, a)
                den = np.real(a.conj().T @ num) # Dénominateur scalaire réel
                w = num / den
            except np.linalg.LinAlgError:
                bmode_mvdr[iz, ix] = 0.0
                continue
            
            # Application
            pixel_val = w.conj().T @ x_vec
            rf_mvdr[iz, ix] = np.real(pixel_val)
            bmode_mvdr[iz, ix] = np.abs(pixel_val)

    # --- 4. Sortie ---
    val_max = np.max(bmode_mvdr)
    if val_max == 0: val_max = 1.0
    
    # Enveloppe linéaire normalisée (CIBLE POUR IA)
    env = bmode_mvdr / val_max
    
    # Image Log (POUR HUMAIN)
    bmode_dB = 20 * np.log10(env + 1e-12)

    meta = {
        'c': c,
        'f0': f0,
        'fs': fs,
        'Nelem': Nelem,
        'pitch': pitch,
        'x_img': x_img,
        'z_img': z_img,
        'SNR_dB': SNR_dB,
    }

    data = {
        'rf': rf,
        'x_img': x_img,
        'z_img': z_img,
        'x_el': x_el,
        'bmode_dB': bmode_dB,
        'env': env,   
        'target_rf': rf_mvdr,  
        'meta': meta,
    }
    return data
