import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import glob
import matplotlib.pyplot as plt

# ==========================================
# 1. Pré-traitement & Chargement
# ==========================================

def get_x_el_from_file(f, Nelem=None):
    """Récupère les positions latérales des éléments."""
    keys = ['x_el', 'x_elements', 'elements', 'x_positions']
    for k in keys:
        if k in f: return np.array(f[k][:])
    if 'meta' in f and 'x_el' in f['meta']: return np.array(f['meta']['x_el'])
    
    if Nelem is None: Nelem = f['rf'].shape[1] if 'rf' in f else 128
    pitch = 0.300e-3 
    aperture = (Nelem - 1) * pitch
    return np.linspace(-aperture / 2, aperture / 2, Nelem)

def interpolation(rf, t_idx):
    """Interpolation linéaire des signaux RF."""
    T, N = rf.shape
    t_idx = np.clip(t_idx, 0, T - 1)
    idx_flat = t_idx.reshape(-1, N)
    out_flat = np.empty_like(idx_flat, dtype=rf.dtype)
    x_axis = np.arange(T)
    for n in range(N):
        out_flat[:, n] = np.interp(idx_flat[:, n], x_axis, rf[:, n])
    return out_flat

def extract_pixel_features(rf, x_el, X, Z, c=1540.0, fs=None):
    if fs is None: fs = 40e6 
    dt = 1/fs
    T, N = rf.shape
    H, W = X.shape
    x_el = x_el.reshape(1, 1, N)   
    X = X.reshape(H, W, 1)         
    Z = Z.reshape(H, W, 1)  

    # Délais pour Onde Plane (Z + dist) ou Focalisé (dist + dist)
    # On suppose Onde Plane ici comme dans votre main.py
    dist = np.sqrt((X - x_el)**2 + Z**2)
    delays = (Z + dist) / c          
    t_idx = delays / dt                        

    # Normalisation des inputs (CRUCIAL pour éviter les images noires)
    # On normalise par le max global approximatif pour mettre à l'échelle [-1, 1]
    features = interpolation(rf, t_idx)
    max_val = np.max(np.abs(rf)) if np.max(np.abs(rf)) > 0 else 1.0
    return features / max_val

def extract_pixels_from_h5_list(paths):
    all_X = []
    all_Y = []
    
    if not paths: return torch.empty((0, 0)), torch.empty((0,))

    print(f"Extraction des données depuis {len(paths)} fichiers...")
    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                rf = f["rf"][:]
                
                if 'x_img' in f:
                    x_vec = f['x_img'][:]
                    z_vec = f['z_img'][:]
                    if len(x_vec) == 2: x_vec = np.linspace(x_vec[0], x_vec[-1], 128)
                    if len(z_vec) == 2: z_vec = np.linspace(z_vec[0], z_vec[-1], 128)
                    X, Z = np.meshgrid(x_vec, z_vec)
                else:
                    X, Z = np.meshgrid(np.linspace(-20e-3, 20e-3, 128), np.linspace(5e-3, 50e-3, 128))

                x_el = get_x_el_from_file(f, rf.shape[1])
                X_pixels = extract_pixel_features(rf, x_el, X, Z)
                
                # --- CORRECTION CIBLE ---
                if 'bmode_dB' in f:
                    # bmode_dB est l'image cible (logarithmique)
                    Y_db = f['bmode_dB'][:].reshape(-1)
                    # On convertit en amplitude LINEAIRE positive
                    # db = 20 * log10(amp)  => amp = 10^(db/20)
                    Y_linear = np.power(10, Y_db / 20.0)
                else:
                    Y_linear = np.zeros(X_pixels.shape[0])

                all_X.append(torch.tensor(X_pixels, dtype=torch.float32))
                all_Y.append(torch.tensor(Y_linear, dtype=torch.float32))
                
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if len(all_X) == 0: return torch.empty((0,0)), torch.empty((0,))
    return torch.cat(all_X, dim=0), torch.cat(all_Y, dim=0)

class ABLEDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ==========================================
# 2. Architecture (ABLE)
# ==========================================

class Antirectifier(nn.Module):
    def forward(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=1)

class ABLE_MLP(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.fc1 = nn.Linear(N, N)
        self.act1 = Antirectifier()
        self.drop1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(2 * N, N // 4)
        self.act2 = Antirectifier()
        self.drop2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(N // 2, N // 4)
        self.act3 = Antirectifier()
        self.drop3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(N // 2, N) # Sortie: Poids
    
    def forward(self, x):
        x = self.drop1(self.act1(self.fc1(x)))
        x = self.drop2(self.act2(self.fc2(x)))
        x = self.drop3(self.act3(self.fc3(x)))
        return self.fc4(x)

# ==========================================
# 3. Loss Modifiée (Magnitude)
# ==========================================

class MagnitudeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_rf, target_envelope, weights):
        # 1. On calcule l'enveloppe de la prédiction (Valeur Absolue)
        # Comme on est en batch sans dimension temporelle, |x| est la meilleure approx de l'enveloppe instantanée
        pred_mag = torch.abs(pred_rf)
        
        # 2. Loss principale : on veut que l'amplitude prédite matche l'amplitude cible
        loss_img = self.mse(pred_mag, target_envelope)
        
        # 3. Unity constraint (Somme des poids ~ 1)
        loss_unity = torch.mean((torch.sum(weights, dim=1) - 1.0) ** 2)
        
        return loss_img + 0.1 * loss_unity

# ==========================================
# 4. Entraînement & Inférence
# ==========================================

def training(args):
    print("--- Entraînement ABLE (Mode Magnitude) ---")
    data_dir = os.path.join(args.data_dir, "h5")
    
    # Priorité aux fichiers MVDR pour apprendre la qualité
    h5_paths = sorted(glob.glob(os.path.join(data_dir, "*mvdr.h5")))
    if not h5_paths: h5_paths = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    
    if not h5_paths:
        print("Erreur: Pas de fichiers .h5 trouvés.")
        return

    # Chargement (Convertit dB -> Linear)
    X_train, Y_train = extract_pixels_from_h5_list(h5_paths)
    print(f"Training data shape: {X_train.shape}")
    
    dataset = ABLEDataset(X_train, Y_train)
    # Batch size plus petit peut aider la convergence
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_elem = X_train.shape[1]
    
    model = ABLE_MLP(N=N_elem).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = MagnitudeLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for rf, target in loader:
            rf, target = rf.to(device), target.to(device)
            
            # Prédiction des poids
            weights = model(rf)
            
            # Beamforming (Somme pondérée) -> Signal RF
            pixel_pred_rf = (weights * rf).sum(dim=1)
            
            # Calcul Loss (comparaison des Magnitudes)
            loss = criterion(pixel_pred_rf, target, weights)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(loader):.6f}")

    os.makedirs("weight", exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'N_elem': N_elem}, "weight/able_model.pth")
    print("Modèle sauvegardé.")

def beamforming(args):
    print("--- Inférence ABLE ---")
    model_path = "weight/able_model.pth"
    if not os.path.exists(model_path):
        print("Erreur: Modèle non trouvé.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    model = ABLE_MLP(N=ckpt['N_elem']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    data_dir = os.path.join(args.data_dir, "h5")
    out_dir = os.path.join(args.data_dir, "able_images")
    os.makedirs(out_dir, exist_ok=True)
    
    h5_paths = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            rf = f["rf"][:]
            # Normalisation à l'inférence (comme à l'entraînement)
            max_val = np.max(np.abs(rf)) if np.max(np.abs(rf)) > 0 else 1.0
            
            x_el = get_x_el_from_file(f, rf.shape[1])
            
            # Grid
            if 'x_img' in f:
                x_vec, z_vec = f['x_img'][:], f['z_img'][:]
                if len(x_vec)==2: x_vec = np.linspace(x_vec[0], x_vec[-1], 128)
                if len(z_vec)==2: z_vec = np.linspace(z_vec[0], z_vec[-1], 128)
            else:
                x_vec = np.linspace(-20e-3, 20e-3, 128)
                z_vec = np.linspace(5e-3, 50e-3, 128)
            
            X, Z = np.meshgrid(x_vec, z_vec)
            X_pixels = extract_pixel_features(rf, x_el, X, Z)
            
            # Normalisation Input
            X_pixels = X_pixels / max_val
            
            inp = torch.tensor(X_pixels, dtype=torch.float32).to(device)
            
            # Inférence
            with torch.no_grad():
                weights = model(inp)
                # On récupère le signal RF beamformé
                rf_sum = (weights * inp).sum(dim=1).cpu().numpy()
            
            # Post-traitement: Enveloppe (Abs) -> Log -> Image
            # Note: Comme on n'a pas la dimension temps pour Hilbert ici, on prend Abs
            env = np.abs(rf_sum)
            env = env.reshape(len(z_vec), len(x_vec))
            
            # Log compression
            img_db = 20 * np.log10(env + 1e-12)
            img_db = img_db - np.max(img_db) # Normalize 0 dB max
            
            plt.figure()
            plt.imshow(img_db, cmap='gray', aspect='auto', vmin=-60, vmax=0,
                       extent=[x_vec[0]*1e3, x_vec[-1]*1e3, z_vec[-1]*1e3, z_vec[0]*1e3])
            plt.title(f"ABLE Result: {os.path.basename(path)}")
            plt.xlabel("Lateral [mm]")
            plt.ylabel("Depth [mm]")
            plt.colorbar(label="dB")
            save_name = os.path.join(out_dir, os.path.basename(path).replace(".h5", ".png"))
            plt.savefig(save_name)
            plt.close()
            print(f"Image générée : {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--beamforming", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    if args.training: training(args)
    elif args.beamforming: beamforming(args)
