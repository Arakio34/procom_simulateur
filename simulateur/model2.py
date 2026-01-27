import os
import glob
import argparse

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# ==========================================
# 1. Pré-traitement & Chargement
# ==========================================

def get_x_el_from_file(f, nelem=None):
    """Récupère les positions latérales des éléments."""
    keys = ["x_el", "x_elements", "elements", "x_positions"]
    for k in keys:
        if k in f:
            return np.array(f[k][:])
    if "meta" in f and "x_el" in f["meta"]:
        return np.array(f["meta"]["x_el"])

    if nelem is None:
        nelem = f["rf"].shape[1] if "rf" in f else 128
    pitch = 0.300e-3
    aperture = (nelem - 1) * pitch
    return np.linspace(-aperture / 2, aperture / 2, nelem)


def interpolation(rf, t_idx):
    """Interpolation linéaire des signaux RF."""
    t_len, n_elem = rf.shape
    t_idx = np.clip(t_idx, 0, t_len - 1)
    idx_flat = t_idx.reshape(-1, n_elem)
    out_flat = np.empty_like(idx_flat, dtype=rf.dtype)
    x_axis = np.arange(t_len)
    for n in range(n_elem):
        out_flat[:, n] = np.interp(idx_flat[:, n], x_axis, rf[:, n])
    return out_flat


def extract_pixel_features(rf, x_el, x_grid, z_grid, c=1540.0, fs=40e6):
    dt = 1 / fs
    x_el = x_el.reshape(1, 1, -1)
    x_grid = x_grid.reshape(x_grid.shape[0], x_grid.shape[1], 1)
    z_grid = z_grid.reshape(z_grid.shape[0], z_grid.shape[1], 1)

    dist = np.sqrt((x_grid - x_el) ** 2 + z_grid**2)
    delays = (z_grid + dist) / c
    t_idx = delays / dt

    return interpolation(rf, t_idx)


def _sample_pixels(x_pixels, y_target, max_pixels=65536, rng=None):
    if max_pixels is None or x_pixels.shape[0] <= max_pixels:
        return x_pixels, y_target
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.choice(x_pixels.shape[0], size=max_pixels, replace=False)
    return x_pixels[idx], y_target[idx]


def extract_pixels_from_h5_list(paths, max_pixels_per_file=65536):
    all_x = []
    all_y = []

    if not paths:
        return torch.empty((0, 0)), torch.empty((0,))

    rng = np.random.default_rng(0)
    print(f"Extraction des données depuis {len(paths)} fichiers...")
    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                rf = f["rf"][:]
                rf_max = np.max(np.abs(rf)) if np.max(np.abs(rf)) > 0 else 1.0

                if "x_img" in f:
                    x_vec = f["x_img"][:]
                    z_vec = f["z_img"][:]
                    if len(x_vec) == 2:
                        x_vec = np.linspace(x_vec[0], x_vec[-1], 128)
                    if len(z_vec) == 2:
                        z_vec = np.linspace(z_vec[0], z_vec[-1], 128)
                    x_grid, z_grid = np.meshgrid(x_vec, z_vec)
                else:
                    x_grid, z_grid = np.meshgrid(
                        np.linspace(-20e-3, 20e-3, 128),
                        np.linspace(5e-3, 50e-3, 128),
                    )

                x_el = get_x_el_from_file(f, rf.shape[1])
                x_pixels = extract_pixel_features(rf, x_el, x_grid, z_grid)
                x_pixels = x_pixels / rf_max

                if "target_rf" in f:
                    target_rf = f["target_rf"][:].reshape(-1)
                    y_target = np.abs(target_rf) / rf_max
                else:
                    y_target = np.zeros(x_pixels.shape[0], dtype=np.float32)

                x_pixels, y_target = _sample_pixels(
                    x_pixels, y_target, max_pixels=max_pixels_per_file, rng=rng
                )

                all_x.append(torch.tensor(x_pixels, dtype=torch.float32))
                all_y.append(torch.tensor(y_target, dtype=torch.float32))
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if len(all_x) == 0:
        return torch.empty((0, 0)), torch.empty((0,))
    return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)


class ABLEDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ==========================================
# 2. Architecture (ABLE)
# ==========================================

class Antirectifier(nn.Module):
    def forward(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=1)


class ABLE_MLP(nn.Module):
    def __init__(self, n_elem):
        super().__init__()
        self.fc1 = nn.Linear(n_elem, n_elem)
        self.act1 = Antirectifier()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(2 * n_elem, n_elem // 2)
        self.act2 = Antirectifier()
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(n_elem, n_elem // 2)
        self.act3 = Antirectifier()
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(n_elem, n_elem)

    def forward(self, x):
        x = self.drop1(self.act1(self.fc1(x)))
        x = self.drop2(self.act2(self.fc2(x)))
        x = self.drop3(self.act3(self.fc3(x)))
        return self.fc4(x)


# ==========================================
# 3. Loss (Magnitude + Unity)
# ==========================================

class MagnitudeUnityLoss(nn.Module):
    def __init__(self, unity_weight=0.5):
        super().__init__()
        self.unity_weight = unity_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred_rf, target_mag, weights):
        pred_mag = torch.abs(pred_rf)
        loss_mag = self.l1(pred_mag, target_mag)
        loss_unity = torch.mean((torch.sum(weights, dim=1) - 1.0) ** 2)
        return 1000*(loss_mag*(1-self.unity_weight) + self.unity_weight * loss_unity) #modification pour matcher la loss du papier.
        #return loss_mag + self.unity_weight * loss_unity  # modification pour matcher la loss du papier.

class MagnitudeUnityLoss_V2(nn.Module):
    def __init__(self, unity_weight=0.3):
        super().__init__()
        self.unity_weight = unity_weight
    def forward(self, pred_rf, target_mag, weights):
        eps = 1e-7  # Pour éviter log(0)

        # Séparation positif / négatif avec garde-fou
        pred_pos = torch.clamp(pred_rf, min=eps)
        pred_neg = torch.clamp(-pred_rf, min=eps)
        target_pos = torch.clamp(target_mag, min=eps)
        target_neg = torch.clamp(-target_mag, min=eps)

        # Calcul des erreurs en Log (MSE stable)
        diff_log_pos = torch.log10(target_pos) - torch.log10(pred_pos)
        diff_log_neg = torch.log10(target_neg) - torch.log10(pred_neg)

        # Utilisation de la moyenne pour que la loss soit indépendante de la taille du tenseur
        loss_mag = 0.5 * torch.mean(diff_log_pos ** 2) + 0.5 * torch.mean(diff_log_neg ** 2)

        # Loss d'unité (somme des poids = 1)
        loss_unity = torch.mean((torch.sum(weights, dim=1) - 1.0) ** 2)

        return loss_mag * (1 - self.unity_weight) + self.unity_weight * loss_unity

class SMSLELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # split positive and negative parts
        pred_pos = torch.relu(pred)
        pred_neg = torch.relu(-pred)

        target_pos = torch.relu(target)
        target_neg = torch.relu(-target)

        # compute log mse on each
        loss_pos = torch.mean((torch.log(pred_pos + self.eps) -
                               torch.log(target_pos + self.eps)) ** 2)

        loss_neg = torch.mean((torch.log(pred_neg + self.eps) -
                               torch.log(target_neg + self.eps)) ** 2)

        return loss_pos + loss_neg

class UnityLoss(nn.Module):
    def forward(self, weights):
        s = torch.sum(weights, dim=1)
        return torch.mean((s - 1.0) ** 2)

class ABLELoss(nn.Module):
    def __init__(self, lambda_unity=1):
        super().__init__()
        self.smsle = SMSLELoss()
        self.unity = UnityLoss()
        self.lambda_unity = lambda_unity

    def forward(self, pred_image, target_image, weights):
        L_img = self.smsle(pred_image, target_image)
        L_unity = self.unity(weights)
        return L_img + self.lambda_unity * L_unity


class SimpleMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_rf, target_mag, weights):

        pred_mag = torch.abs(pred_rf)


        return self.mse(pred_mag, target_mag)
# ==========================================
# 4. Entraînement & Inférence
# ==========================================

def training(args):
    print("--- Entraînement ABLE (Mode Magnitude) ---")
    data_dir = os.path.join(args.data_dir, "h5")

    h5_paths = sorted(glob.glob(os.path.join(data_dir, "*mvdr.h5")))
    if not h5_paths:
        h5_paths = sorted(glob.glob(os.path.join(data_dir, "*.h5")))

    if not h5_paths:
        print("Erreur: Pas de fichiers .h5 trouvés.")
        return

    x_train, y_train = extract_pixels_from_h5_list(h5_paths)
    print(f"Training data shape: {x_train.shape}")

    dataset = ABLEDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)
    #loader = DataLoader(dataset, batch_size=16384, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_elem = x_train.shape[1]

    model = ABLE_MLP(n_elem=n_elem).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = MagnitudeUnityLoss()
    #criterion = ABLELoss()
    #criterion = SimpleMSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for rf, target in loader:
            rf, target = rf.to(device), target.to(device)
            weights = model(rf)
            pixel_pred_rf = (weights * rf).sum(dim=1)
            loss = criterion(pixel_pred_rf, target, weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {total_loss/len(loader):.6f}")

    os.makedirs("weight", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "N_elem": n_elem}, "weight/able_model.pth")
    print("Modèle sauvegardé.")


def beamforming(args):
    print("--- Inférence ABLE ---")
    model_path = "weight/able_model.pth"
    if not os.path.exists(model_path):
        print("Erreur: Modèle non trouvé.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    model = ABLE_MLP(n_elem=ckpt["N_elem"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data_dir = os.path.join(args.data_dir, "h5")
    out_dir = os.path.join(args.data_dir, "able_images")
    os.makedirs(out_dir, exist_ok=True)

    h5_paths = sorted(glob.glob(os.path.join(data_dir, "*.h5")))

    for path in h5_paths:
        with h5py.File(path, "r") as f:
            rf = f["rf"][:]
            rf_max = np.max(np.abs(rf)) if np.max(np.abs(rf)) > 0 else 1.0

            x_el = get_x_el_from_file(f, rf.shape[1])

            if "x_img" in f:
                x_vec, z_vec = f["x_img"][:], f["z_img"][:]
                if len(x_vec) == 2:
                    x_vec = np.linspace(x_vec[0], x_vec[-1], 128)
                if len(z_vec) == 2:
                    z_vec = np.linspace(z_vec[0], z_vec[-1], 128)
            else:
                x_vec = np.linspace(-20e-3, 20e-3, 128)
                z_vec = np.linspace(5e-3, 50e-3, 128)

            x_grid, z_grid = np.meshgrid(x_vec, z_vec)
            x_pixels = extract_pixel_features(rf, x_el, x_grid, z_grid)
            x_pixels = x_pixels / rf_max

            inp = torch.tensor(x_pixels, dtype=torch.float32).to(device)

            with torch.no_grad():
                weights = model(inp)
                rf_sum = (weights * inp).sum(dim=1).cpu().numpy()

            env = np.abs(rf_sum).reshape(len(z_vec), len(x_vec))
            img_db = 20 * np.log10(env + 1e-12)
            img_db = img_db - np.max(img_db)

            plt.figure()
            plt.imshow(
                img_db,
                cmap="gray",
                aspect="auto",
                vmin=-60,
                vmax=0,
                extent=[x_vec[0] * 1e3, x_vec[-1] * 1e3, z_vec[-1] * 1e3, z_vec[0] * 1e3],
            )
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

    if args.training:
        training(args)
    elif args.beamforming:
        beamforming(args)
