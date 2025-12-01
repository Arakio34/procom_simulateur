import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.io import savemat
from scipy.signal.windows import hann


def generate_bright_point(seed,
                          max_point,
                          x_range=(-10e-3, 10e-3),
                          z_range=(10e-3, 50e-3),
                          amp_range=(1.0, 3.0)):

    rng = np.random.default_rng(seed)
    nb_points = round(np.random.rand()*9 % max_point)+1
    bright_points = np.zeros((nb_points,3))
    for nb in range(nb_points):
        bright_points[nb][0] = rng.uniform(x_range[0], x_range[1])
        bright_points[nb][1] = rng.uniform(z_range[0], z_range[1])
        bright_points[nb][2] = round(rng.uniform(amp_range[0], amp_range[1]) )
    print(bright_points)
    return bright_points
        

def simulate_us_scene(
    N_speckle=0,
    SNR_dB=10.0,
    seed=None,
    save_mat_path=None,
    save_png_path=None,
    plot=False,
    max_point=3,
):
    """
    Simule une image d'échographie B-mode et les RF associés.

    Retourne un dict avec rf, bmode_dB, env, meta, etc.
    """

    # ============================
    # Graine aléatoire
    # ============================
    if seed is not None:
        print(seed)
        #np.random.seed(seed)

    # ============================
    # Paramètres de base
    # ============================
    c        = 1540.0
    f0       = 5e6
    fracBW   = 0.6
    fs       = 40e6
    lam      = c / f0
    dt       = 1.0 / fs  # pas vraiment utilisé mais bon

    # ============================
    # Array
    # ============================
    Nelem    = 80
    pitch    = 0.3e-3
    aperture = (Nelem - 1) * pitch
    x_el     = np.linspace(-aperture / 2, aperture / 2, Nelem)
    z_el     = 0.0  # non utilisé ensuite

    # ============================
    # Grille image
    # ============================
    x_span   = 20e-3
    z_min    = 10e-3
    z_max    = 50e-3
    Nx       = 256
    Nz       = 256
    x_img    = np.linspace(-x_span / 2, x_span / 2, Nx)
    z_img    = np.linspace(z_min, z_max, Nz)

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
    # Speckle + points brillants
    # ============================
    xs = ROI_x[0] + (ROI_x[1] - ROI_x[0]) * np.random.rand(N_speckle)
    zs = ROI_z[0] + (ROI_z[1] - ROI_z[0]) * np.random.rand(N_speckle)
    as_ = rayleighScale * np.abs(
        (np.random.randn(N_speckle) + 1j * np.random.randn(N_speckle)) / np.sqrt(2.0)
    )

    bright_points = generate_bright_point(seed,max_point)

    xs = np.concatenate([xs, bright_points[:, 0]])
    zs = np.concatenate([zs, bright_points[:, 1]])
    as_ = np.concatenate([as_, bright_points[:, 2]])

    N_scatt = as_.size

    # ============================
    # Synthèse RF
    # ============================
    rf = np.zeros((Nt, Nelem), dtype=np.float32)

    for n in range(Nelem):
        dx_n = xs - x_el[n]
        Rrx  = np.sqrt(dx_n ** 2 + zs ** 2)
        t_tx = zs / c
        t_rx = Rrx / c
        tau  = t_tx + t_rx

        att  = 1.0 / np.maximum(Rrx, 1e-3)

        sig_n = np.zeros(Nt, dtype=np.float32)
        for k in range(N_scatt):
            tk = tau[k]
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

    # ============================
    # Beamforming DAS
    # ============================
    Xg, Zg = np.meshgrid(x_img, z_img, indexing='xy')  # pas vraiment utilisé

    bmode_lin = np.zeros((Nz, Nx), dtype=np.float32)
    y_align   = np.zeros((Nelem, Nx, Nz), dtype=np.float32)

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
    analytic = hilbert(bmode_lin, axis=0)
    env      = np.abs(analytic)
    env_max  = np.max(env + eps)
    env      = env / env_max
    bmode_dB = 20 * np.log10(env + eps)

    # ============================
    # Affichage / sauvegarde PNG
    # ============================
    if plot or (save_png_path is not None):
        plt.figure()
        plt.imshow(
            bmode_dB,
            extent=[x_img[0] * 1e3, x_img[-1] * 1e3, z_img[-1] * 1e3, z_img[0] * 1e3],
            cmap='gray',
            aspect='equal'
        )
        plt.clim(-60, 0)
        plt.xlabel('x [mm]')
        plt.ylabel('z [mm]')
        plt.title('B-mode (DAS, dB)')
        plt.colorbar(label='dB')

        if save_png_path is not None:
            plt.savefig(save_png_path, dpi=300, bbox_inches='tight')

        if plot:
            plt.show()
        else:
            plt.close()

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
        'phantom': {
            'xs': xs,
            'zs': zs,
            'as': as_,
            'bright_point': {
                'pos': pt_pos,
                'amp': pt_amp
            }
        }
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

    if save_mat_path is not None:
        save_h5(save_mat_path, data)
        print(f'Saved: {save_mat_path}')

    return data



def save_h5(path, data):
    """
    Sauvegarde un dictionnaire Python dans un fichier HDF5 générique.
    Gère les scalaires, arrays, et dictionnaires imbriqués.
    """

    def write_group(h5group, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                subgroup = h5group.create_group(key)
                write_group(subgroup, value)
            else:
                h5group.create_dataset(key, data=value)

    with h5py.File(path, "w") as f:
        write_group(f, data)

    print(f"[OK] Saved HDF5 → {path}")

# ============================
# Exemple : générer 10 scènes
# ============================
if __name__ == "__main__":
    for i in range(10):
        mat_path = f"sample_{i:03d}.h5"
        png_path = f"bmode_{i:03d}.png"

        simulate_us_scene(
            N_speckle=0,        
            SNR_dB=+15.0,
            seed=i,               
            save_mat_path=mat_path,
            save_png_path=png_path,
            plot=False,           
        )

