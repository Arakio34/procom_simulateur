import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.linalg import inv, pinv
from scipy.io import savemat
from scipy.signal.windows import hann
import argparse
import os
# TODO
# Changement du comportement : 
# - Desctiption de l'environnement soit via Json soit via terminal
# - Pouvoire choisire une desciption total des cibles via Json
# - Soit aleatoire avec une seed pour les cible

# Changement d'architecture :
# - Description de de la scene en Json
# - Description de des cibles en Json
#
# wrapper json -> objet cibles 
# wrapper json -> objet scene 


def generate_bright_point(
    seed,
    max_point,
    x_range=(-10e-3, 10e-3), 
    z_range=(10e-3, 50e-3),
    amp_range=(1.0, 3.0)
):

    rng = np.random.default_rng(seed)
    nb_points = round(np.random.rand()*9 % max_point)+1
    bright_points = np.zeros((nb_points,3))
    for nb in range(nb_points):
        bright_points[nb][0] = rng.uniform(x_range[0], x_range[1])
        bright_points[nb][1] = rng.uniform(z_range[0], z_range[1])
        bright_points[nb][2] = round(rng.uniform(amp_range[0], amp_range[1]) )
    return bright_points
        
# bp List au format [[x1,z1,i1],[x2,z2,i2],...,[xn,zn,in]]
def simulate_us_scene(
    SNR_dB=10.0,
    Nelem=80,
    seed=None,
    max_point=3,
    generate_bp=True,
    bp_list=None
):
    """
    Simule une image d'échographie B-mode et les RF associés.
    Retourne un dict avec rf, bmode_dB, env, meta, etc.
    """

    x_span   = 20e-3
    z_min    = 10e-3
    z_max    = 50e-3
    Nx       = 256
    Nz       = 256
    x_img    = np.linspace(-x_span / 2, x_span / 2, Nx)
    z_img    = np.linspace(z_min, z_max, Nz)

    # ============================
    # Graine aléatoire
    # ============================
    if seed is not None:
        np.random.seed(seed)

    # ============================
    # Paramètres de base
    # ============================
    c        = 1540.0
    f0       = 5e6
    fracBW   = 0.6
    fs       = 40e6
    lam      = c / f0
    dt       = 1.0 / fs  

    # ============================
    # Array
    # ============================
    pitch    = 0.15e-3
    aperture = (Nelem - 1) * pitch
    x_el     = np.linspace(-aperture / 2, aperture / 2, Nelem)


    # ============================
    # Phantom
    # ============================
    ROI_x         = np.array([-x_span / 2, x_span / 2])
    ROI_z         = np.array([z_min, z_max])
    rayleighScale = 1.0

    pt_pos = np.array([0e-3, 30e-3])
    pt_amp = 8.0

    # ============================
    # Pulse
    # ============================
    nCycles = 2.5
    pulseT  = nCycles / f0
    t_pulse = np.arange(-pulseT, pulseT + 1.0 / fs, 1.0 / fs)
    sigma_t = pulseT / 2.355
    pulse   = np.cos(2 * np.pi * f0 * t_pulse) * np.exp(-(t_pulse ** 2) / (2 * sigma_t ** 2))

    # ============================
    # Temps max
    # ============================
    z_max_toa = z_max / c
    r_max     = np.sqrt((x_span / 2 + aperture / 2) ** 2 + z_max ** 2)
    t_max     = z_max_toa + r_max / c + 2 / f0
    t         = np.arange(0.0, t_max + 1.0 / fs, 1.0 / fs)
    Nt        = t.size

    # ============================
    # Apodisation RX
    # ============================
    use_hann = 1
    if use_hann == 1:
        apo_rx = hann(Nelem)
    else:
        apo_rx = np.ones(Nelem)

    # ============================
    #  Points brillants
    # ============================

    if generate_bp == True:
        bright_points = generate_bright_point(seed,max_point)
    else:
        bright_points = bp_list


    xs = bright_points[:, 0]
    zs = bright_points[:, 1]
    as_ = bright_points[:, 2]

    print(xs)
    print(zs)
    print(as_)

    N_scatt = as_.size

    # ============================
    #  Couche absorbante
    # ============================

    #   [[zmin1,zmax2,v_1],
    #    [zmin2,zmax2,v_2],
    #  ...
    #    ]
    # avec v_i le coeficient de reflexion

    couches = np.array([[20e-3,40e-3,0.40]])
    #Juste pour le fun de fair un droite
    droite = np.linspace(0,10e-3,Nelem)
    for n in range(couches.shape[0]):
        #Debut de la couche
        xs = np.append(xs,x_el.reshape(-1,1))
        zs = np.append(zs,np.full(Nelem,couches[n][0])+droite)
        as_ = np.append(as_,np.full(Nelem,couches[n][2]))

        #Fin de la couche
        xs = np.append(xs,x_el.reshape(-1,1))
        zs = np.append(zs,np.full(Nelem,couches[n][1]))
        as_ = np.append(as_,np.full(Nelem,couches[n][2]**2))
    #TODO ! En utilisant la methode diffuseur, chaque capteur prend en compte l'onde reflechis a chaque point de l'interface
    # entre le milieu 1 et 2. Cella pose probleme etant donner que seul le capteur en direct resoit l'onde
    
    # ============================
    # Synthèse RF
    # ============================
    rf = np.zeros((Nt, Nelem), dtype=np.float32)

    for n in range(Nelem):
        #Calcule de la distance sur l'axe des x 
        dx_n = xs - x_el[n]
        #Calcule de la distance radial
        Rrx  = np.sqrt(dx_n ** 2 + zs ** 2)
        #Temps de propagation aller (onde plane)
        t_tx = zs / c
        #Temps de propagation retour 
        t_rx = Rrx / c
        #Retard total
        tau  = t_tx + t_rx

        #Attenuation en 1/R avec un np.maximum pour eviter les division par zeros
        att  = 1.0 / np.maximum(Rrx, 1e-3)

        sig_n = np.zeros(Nt, dtype=np.float32)

        for k in range(as_.shape[0]):
            # Retard du PB k par rapport au capteur n
            tk = tau[k]
            # Amplitude par rapport au niveau de reflexion [as] et a l'attenuation [att]
            ak = as_[k] * att[k]
            sig_n += np.float32(
                ak * np.interp(t, t_pulse + tk, pulse, left=0.0, right=0.0)
            )
        rf[:, n] = sig_n

    # ============================
    # Bruit
    # ============================
    eps = np.finfo(np.float32).eps
    signal_pow = np.mean(rf.astype(np.float64) ** 2 + eps)
    noise_pow  = signal_pow / (10 ** (SNR_dB / 10))
    rf = rf + np.sqrt(noise_pow).astype(np.float32) * np.random.randn(*rf.shape).astype(np.float32)
    return rf
