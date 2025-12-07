import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.linalg import inv, pinv
from scipy.io import savemat
from scipy.signal.windows import hann
import argparse
import os

def beamforming(
    rf,
    Nelem=80,
    save_path=None,
    SNR_dB=10.0,
    plot=False,):

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

def mvdr_beamforming(rf, Nelem=80, SNR_dB=10.0, regularization=0.1):
    """
    Implémentation simplifiée du MVDR/Capon
    """
    # ... (Reprendre les mêmes paramètres de grille que votre fonction beamforming) ...
    c, f0, fs = 1540.0, 5e6, 40e6
    x_span, z_min, z_max = 20e-3, 10e-3, 50e-3
    Nx, Nz = 128, 128 # Réduit pour l'exemple (256x256 est très lent en MVDR pur CPU)
    
    # 1. Alignement Temporel (Exactement comme dans votre DAS)
    # On recalcule y_align (les signaux alignés sur chaque pixel)
    # ... (Copier la logique de calcul de y_align de votre fonction beamforming) ...
    # Supposons qu'on a récupéré y_align de forme (Nelem, Nx, Nz)
    
    # Pour le MVDR, il est préférable de travailler sur le signal Analytique (Complexe)
    # pour avoir l'information de phase correcte lors du calcul de covariance.
    y_align_analytic = hilbert(y_align, axis=0) # Hilbert sur l'axe temporel (capteurs) ? 
    # Non, attention : Hilbert doit être fait sur le signal RF brut ou temporellement.
    # Ici, pour simplifier, on suppose y_align complexe ou on le fait pixel par pixel.
    
    # Structure de sortie
    bmode_mvdr = np.zeros((Nz, Nx), dtype=np.float32)
    
    # Vecteur de direction (Steering vector) : Comme tout est aligné, on cherche la somme
    # donc a est un vecteur de 1.
    a = np.ones((Nelem, 1), dtype=np.complex64)
    
    # Facteur de "Diagonal Loading" pour rendre l'inversion stable (indispensable)
    # On ajoute un petit bruit fictif sur la diagonale de la matrice R
    delta = regularization * np.trace(np.eye(Nelem)) / Nelem 

    print("Calcul MVDR en cours (patience...)...")
    
    # Boucle sur les pixels 
    for iz in range(Nz):
        for ix in range(Nx):
            # 1. Récupérer le vecteur de données pour ce pixel (Snapshot)
            # Forme : (Nelem, 1)
            x_vec = y_align_analytic[:, ix, iz].reshape(-1, 1)
            
            # 2. Calculer la Matrice de Covariance R
            # En pratique, on moyenne R sur une petite fenêtre spatiale (Spatial Smoothing)
            # pour avoir une stat robuste. Ici, on fait du "Snapshot MVDR" simple :
            R = x_vec @ x_vec.conj().T
            
            # Ajout du Diagonal Loading (Régularisation)
            R = R + delta * np.eye(Nelem)
            
            # 3. Calcul des poids MVDR : w = (R^-1 * a) / (a^H * R^-1 * a)
            # On utilise pinv ou solve pour la stabilité
            try:
                R_inv = np.linalg.inv(R)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R)
                
            num = R_inv @ a
            den = a.conj().T @ num
            w = num / den
            
            # 4. Appliquer les poids (Beamforming)
            pixel_val = w.conj().T @ x_vec
            
            # On stocke le module (l'énergie)
            bmode_mvdr[iz, ix] = np.abs(pixel_val)

    # Conversion en dB
    bmode_dB = 20 * np.log10(bmode_mvdr / np.max(bmode_mvdr) + 1e-12)
    return bmode_dB
