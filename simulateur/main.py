import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.linalg import inv, pinv
from scipy.io import savemat
from scipy.signal.windows import hann
import argparse
import os
from utils import save_h5, save_image
from simulateur import simulate_us_scene
from beamforming import beamforming, mvdr_beamforming

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulateur Echographique B-Mode")
    
    # Arguments demandés précédemment
    parser.add_argument("--num", type=int, default=10, help="Nombre d'images à générer")
    parser.add_argument("--out", type=str, default="data", help="Dossier de sortie racine")
    parser.add_argument("--show", action="store_true", help="Afficher les plots (bloquant)")
    parser.add_argument("--snr", type=float, default=15.0, help="Rapport Signal/Bruit désiré en dB.")
    parser.add_argument("--nelem", type=int, default=80, help="Nombre de capteur.")
    parser.add_argument("--mvdr", type=bool, default=False, help="Activation du MVDR.")
    parser.add_argument("--maxpoint", type=int, default=3, help="Nombre de point maximum par image.")
    
    args = parser.parse_args()

    h5_dir = os.path.join(args.out, "h5")
    img_dir = os.path.join(args.out, "images")
    
    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    print(f"--- Démarrage de la simulation ---")
    print(f"Nombre d'images : {args.num}")
    print(f"SNR (dB) : {args.snr}")
    print(f"Dossier sortie  : {args.out}")

    for i in range(args.num):
        filename_base = f"sample_{i:04d}" 
        
        h5_path = os.path.join(h5_dir, f"{filename_base}.h5")

        
        rf = simulate_us_scene(
            SNR_dB=args.snr,         
            Nelem=args.nelem,         
            seed=i,               
            max_point=args.maxpoint
        )

        if args.mvdr == True:
            png_path = os.path.join(img_dir, f"{filename_base}_mvdr.png")
            data = mvdr_beamforming( 
                rf, 
                Nelem = args.nelem,
                SNR_dB=args.snr,         
            )
        else:
            png_path = os.path.join(img_dir, f"{filename_base}.png")
            data = beamforming( 
                rf, 
                Nelem = args.nelem,
                SNR_dB=args.snr,         
            )

        if h5_path is not None:
            save_h5(h5_path, data)
            save_image(png_path, data)
            print(f'Saved: {h5_path}')
    
    print("\n--- Terminé ! ---")# ============================
