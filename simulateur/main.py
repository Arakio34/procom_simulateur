import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.linalg import inv, pinv
from scipy.io import savemat
from scipy.signal.windows import hann
import argparse
import os
from utils import save_h5, save_image, load_scene_data # Assure-toi d'avoir renommé/mis à jour utils.py
from simulateur import simulate_us_scene
from beamforming import beamforming, mvdr_beamforming

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulateur Echographique B-Mode")
    
    parser.add_argument(
        '--json-file',
        type=load_scene_data, 
        help="Chemin vers le fichier JSON contenant 'points' et optionnellement 'layers'."
    )
    parser.add_argument("--num", type=int, default=10, help="Nombre d'images à générer (ignoré si JSON unique, sauf si forcé).")
    parser.add_argument("--out", type=str, default="data", help="Dossier de sortie racine")
    parser.add_argument("--snr", type=float, default=15.0, help="Rapport Signal/Bruit désiré en dB.")
    parser.add_argument("--nelem", type=int, default=80, help="Nombre de capteur.")
    parser.add_argument("--mvdr", type=bool, default=False, help="Activation du MVDR.")
    parser.add_argument("--maxpoint", type=int, default=3, help="Nombre de point maximum par image (mode aléatoire).")
    
    args = parser.parse_args()
    
    # --- Gestion des données d'entrée (JSON vs Aléatoire) ---
    scene_data = args.json_file # Ceci est maintenant un dictionnaire ou None
    
    if scene_data is not None:
        print(f"Chargement de la scène depuis le fichier JSON...")
        points = np.array(scene_data['points'])
        layers = scene_data['layers'] # Liste de dicts ou liste vide
        
        # Affichage pour contrôle
        print(f"  - {len(points)} cibles détectées.")
        if len(layers) > 0:
            print(f"  - {len(layers)} couches détectées.")
            for l in layers:
                print(f"    * {l.get('name', 'Couche')} : {l['z_min']*1e3:.1f}-{l['z_max']*1e3:.1f}mm (c={l['c']} m/s)")
        else:
            print(f"  - Aucune couche définie (Milieu homogène).")

        rand_gen = False
        num = 1 # Par défaut, une scène statique = 1 image. Tu peux mettre 'args.num' si tu veux simuler du bruit plusieurs fois.
    else:
        print("Mode génération aléatoire.")
        points = None
        layers = [] # Pas de couches en mode aléatoire (ou tu pourrais en générer ici)
        rand_gen = True
        num = args.num

    # --- Préparation des dossiers de sortie ---
    h5_dir = os.path.join(args.out, "h5")
    img_dir = os.path.join(args.out, "images")
    
    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    print(f"--- Démarrage de la simulation ---")
    print(f"Nombre d'images à générer : {num}")
    print(f"SNR (dB) : {args.snr}")
    print(f"Dossier sortie  : {args.out}")

    # --- Boucle de simulation ---
    for i in range(num):
        filename_base = f"sample_{i:04d}" 
        h5_path = os.path.join(h5_dir, f"{filename_base}.h5")

        # Appel avec le nouvel argument layers_list
        rf = simulate_us_scene(
            SNR_dB=args.snr,         
            Nelem=args.nelem,         
            seed=i,               
            max_point=args.maxpoint,
            generate_bp = rand_gen,
            bp_list = points,
            layers_list = layers  # <--- Ajout des couches ici
        )

        # Beamforming (Reconstruction)
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

        # Sauvegarde
        if h5_path is not None:
            # On ajoute les infos des couches dans les métadonnées sauvegardées si pertinent
            if layers:
                data['meta']['layers'] = str(layers) # Conversion str simple pour le h5
            
            save_h5(h5_path, data)
            save_image(png_path, data)
            print(f'Saved: {h5_path}')
    
    print("\n--- Terminé ! ---")
