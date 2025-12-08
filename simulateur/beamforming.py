import numpy as np
import h5py
from scipy.signal import hilbert
from scipy.linalg import inv, pinv
from scipy.io import savemat
from scipy.signal.windows import hann
import argparse
import os

def beamforming(
    rf,
    Nelem=80,
    SNR_dB=10.0,):

    c        = 1540.0
    f0       = 5e6
    fracBW   = 0.6
    fs       = 40e6
    lam      = c / f0
    dt       = 1.0 / fs  

    x_span   = 20e-3
    z_min    = 10e-3
    z_max    = 50e-3
    Nx       = 256
    Nz       = 256
    x_img    = np.linspace(-x_span / 2, x_span / 2, Nx)
    z_img    = np.linspace(z_min, z_max, Nz)

    Xg, Zg = np.meshgrid(x_img, z_img, indexing='xy')  

    bmode_lin = np.zeros((Nz, Nx), dtype=np.float32)
    y_align   = np.zeros((Nelem, Nx, Nz), dtype=np.float32)

    pitch    = 0.15e-3
    aperture = (Nelem - 1) * pitch
    x_el     = np.linspace(-aperture / 2, aperture / 2, Nelem)

    z_max_toa = z_max / c
    r_max     = np.sqrt((x_span / 2 + aperture / 2) ** 2 + z_max ** 2)
    t_max     = z_max_toa + r_max / c + 2 / f0
    t         = np.arange(0.0, t_max + 1.0 / fs, 1.0 / fs)
    Nt        = t.size

    use_hann = 1
    if use_hann == 1:
        apo_rx = hann(Nelem)
    else:
        apo_rx = np.ones(Nelem)

    eps = np.finfo(np.float32).eps

    # ============================
    # Pulse
    # ============================
    nCycles = 2.5
    pulseT  = nCycles / f0
    t_pulse = np.arange(-pulseT, pulseT + 1.0 / fs, 1.0 / fs)
    sigma_t = pulseT / 2.355
    pulse   = np.cos(2 * np.pi * f0 * t_pulse) * np.exp(-(t_pulse ** 2) / (2 * sigma_t ** 2))
    for ix in range(Nx):
        x0 = x_img[ix]
        zz = z_img
        for n in range(Nelem):
            dx = x0 - x_el[n]
            Rrx = np.sqrt(dx ** 2 + zz ** 2)
            tau_tot = (zz / c) + (Rrx / c)
            y_n = np.interp(tau_tot, t, rf[:, n], left=0.0, right=0.0)
            y_align[n, ix, :] = y_n.astype(np.float32)

        tmp  = y_align[:, ix, :]     # (Nelem, Nz)
        tmp2 = apo_rx[:, None] * tmp
        y_sum = np.sum(tmp2, axis=0)  # (Nz,)
        bmode_lin[:, ix] = y_sum.astype(np.float32)

    # ============================
    # Enveloppe + log-compression 
    # ============================

    # 1. On définit une taille de padding (ex: 64 pixels de marge)
    pad_width = 64

    # 2. On ajoute des zéros en haut et en bas de l'axe 0 (profondeur z)
    # bmode_lin est de forme (Nz, Nx)
    bmode_padded = np.pad(bmode_lin, ((pad_width, pad_width), (0, 0)), mode='constant')

    # 3. On applique Hilbert sur la version allongée
    analytic_padded = hilbert(bmode_padded, axis=0)

    # 4. On retire le padding pour retrouver la taille originale (Nz, Nx)
    analytic = analytic_padded[pad_width:-pad_width, :]

    env      = np.abs(analytic)
    env_max  = np.max(env + eps)
    env      = env / env_max
    bmode_dB = 20 * np.log10(env + eps)


    # ============================
    # Meta + sauvegarde .h5
    # ============================
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
        'nCycles': nCycles,
        'fracBW': fracBW,
    }

    data = {
        'rf': rf,
        't': t,
        'x_el': x_el,
        'x_img': x_img,
        'z_img': z_img,
        'bmode_dB': bmode_dB,
        'env': env,
        'y_align': y_align,
        'meta': meta,
    }
    return data

#Fonction mvdr_beamforming generé a l'aide de gemini AI [Google]
def mvdr_beamforming(rf, Nelem=80, SNR_dB=10.0, regularization=0.1):
    """
    Implémentation Corrigée du MVDR/Capon
    """
    c, f0, fs = 1540.0, 5e6, 40e6
    x_span, z_min, z_max = 20e-3, 10e-3, 50e-3
    
    # On réduit la résolution pour que le calcul MVDR (très lourd) ne prenne pas 1 heure
    # Tu peux remonter à 256 si tu es patient
    Nx, Nz = 64, 64 
    
    x_img    = np.linspace(-x_span / 2, x_span / 2, Nx)
    z_img    = np.linspace(z_min, z_max, Nz)

    # --- 1. Préparation des données (Crucial pour MVDR) ---
    # On passe en complexe (Analytique) TOUT DE SUITE pour garder la phase
    rf_analytic = hilbert(rf, axis=0) 

    bmode_mvdr = np.zeros((Nz, Nx), dtype=np.float32)
    
    # Matrice pour stocker les signaux alignés (Complexe maintenant !)
    y_align   = np.zeros((Nelem, Nx, Nz), dtype=np.complex64)

    pitch    = 0.15e-3
    aperture = (Nelem - 1) * pitch
    x_el     = np.linspace(-aperture / 2, aperture / 2, Nelem)

    # Axe temporel
    z_max_toa = z_max / c
    r_max     = np.sqrt((x_span / 2 + aperture / 2) ** 2 + z_max ** 2)
    t_max     = z_max_toa + r_max / c + 2 / f0
    t         = np.arange(0.0, t_max + 1.0 / fs, 1.0 / fs)
    
    # Pulse pour reference (optionnel ici mais gardé pour cohérence)
    pulseT  = 2.5 / f0

    print("1/2 : Alignement des signaux (Delay)...")
    
    # --- 2. Phase d'Alignement (Delay) ---
    # C'est la partie qui manquait !
    for ix in range(Nx):
        x0 = x_img[ix]
        # On vectorise sur Z pour aller un peu plus vite
        for n in range(Nelem):
            dx = x0 - x_el[n]
            # Distance pour tous les points Z d'une colonne
            Rrx = np.sqrt(dx ** 2 + z_img ** 2) 
            tau_tot = (z_img / c) + (Rrx / c)
            
            # Interpolation complexe (Reel et Imag séparés car numpy ne gère pas interp complexe direct)
            val_real = np.interp(tau_tot, t, np.real(rf_analytic[:, n]), left=0.0, right=0.0)
            val_imag = np.interp(tau_tot, t, np.imag(rf_analytic[:, n]), left=0.0, right=0.0)
            
            y_align[n, ix, :] = val_real + 1j * val_imag

    print("2/2 : Calcul des poids MVDR (Inversion Matrice)...")

    # --- 3. Phase MVDR (Calcul des poids w) ---
    # Vecteur de direction (Steering vector) : Comme on a déjà aligné les signaux,
    # on veut juste sommer en phase -> vecteur de 1.
    a = np.ones((Nelem, 1), dtype=np.complex64)
    
    for iz in range(Nz):
        for ix in range(Nx):
            # Snapshot : le vecteur signal reçu par les N capteurs pour ce pixel
            # Forme (Nelem, 1)
            x_vec = y_align[:, ix, iz].reshape(-1, 1)
            
            # Calcul de la Covariance Spatiale R = x * x^H
            # (Note: En vrai MVDR robuste, on fait du "Spatial Smoothing" ici, 
            # c'est-à-dire qu'on moyenne R sur des sous-réseaux, mais commençons simple)
            R = x_vec @ x_vec.conj().T
            
            # Diagonal Loading (Indispensable pour inverser R qui est de rang 1 ici)
            # On prend un % de la puissance moyenne (trace)
            power = np.trace(R).real
            if power < 1e-12: 
                # Si le pixel est vide (bruit pur), on met 0 et on passe
                bmode_mvdr[iz, ix] = 0.0
                continue
                
            delta = (regularization * power / Nelem) 
            R_loaded = R + delta * np.eye(Nelem)
            
            # Calcul des poids w = (R^-1 * a) / (a^H * R^-1 * a)
            try:
                # solve est plus rapide et stable que inv
                # On résout R * num = a  => num = R^-1 * a
                num = np.linalg.solve(R_loaded, a)
            except np.linalg.LinAlgError:
                # Fallback si singulier
                num = np.linalg.pinv(R_loaded) @ a
                
            den = a.conj().T @ num
            w = num / den
            
            # Application du poids (Beamforming)
            pixel_val = w.conj().T @ x_vec
            
            bmode_mvdr[iz, ix] = np.abs(pixel_val)

    # Conversion en dB
    # On ajoute epsilon pour éviter log(0)
    val_max = np.max(bmode_mvdr)
    if val_max == 0: val_max = 1.0 # Sécurité
    
    bmode_dB = 20 * np.log10(bmode_mvdr / val_max + 1e-12)

    # Structure de retour identique à ta fonction précédente
    meta = {
        'c': c, 'f0': f0, 'fs': fs, 'Nelem': Nelem,
        'pitch': pitch, 'x_img': x_img, 'z_img': z_img,
    }

    data = {
        'rf': rf,
        'x_img': x_img,
        'z_img': z_img,
        'bmode_dB': bmode_dB,
        'meta': meta,
    }
    return data
