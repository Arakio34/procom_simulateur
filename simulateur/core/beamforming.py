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
    nx = params.Nx
    nz = params.Nz
    nelem = params.Nelem if nelem is None else nelem
    snr_db = params.SNR_dB if snr_db is None else snr_db

    x_img = np.linspace(-x_span / 2, x_span / 2, nx)
    z_img = np.linspace(z_min, z_max, nz)

    bmode_lin = np.zeros((nz, nx), dtype=np.float32)
    y_align = np.zeros((nelem, nx, nz), dtype=np.float32)

    pitch = params.pitch
    aperture = (nelem - 1) * pitch
    x_el = np.linspace(-aperture / 2, aperture / 2, nelem)

    z_max_toa = z_max / c
    r_max = np.sqrt((x_span / 2 + aperture / 2) ** 2 + z_max**2)
    t_max = z_max_toa + r_max / c + 2 / f0
    t = np.arange(0.0, t_max + 1.0 / fs, 1.0 / fs)

    apo_rx = hann(nelem)
    eps = np.finfo(np.float32).eps

    for ix in range(nx):
        x0 = x_img[ix]
        zz = z_img
        for n in range(nelem):
            dx = x0 - x_el[n]
            rrx = np.sqrt(dx**2 + zz**2)
            tau_tot = (zz / c) + (rrx / c)
            y_n = np.interp(tau_tot, t, rf[:, n], left=0.0, right=0.0)
            y_align[n, ix, :] = y_n.astype(np.float32)

        tmp = y_align[:, ix, :]
        tmp2 = apo_rx[:, None] * tmp
        y_sum = np.sum(tmp2, axis=0)
        bmode_lin[:, ix] = y_sum.astype(np.float32)

    pad_width = 64
    bmode_padded = np.pad(
        bmode_lin, ((pad_width, pad_width), (0, 0)), mode="constant"
    )
    analytic_padded = hilbert(bmode_padded, axis=0)
    analytic = analytic_padded[pad_width:-pad_width, :]

    env = np.abs(analytic)
    env_max = np.max(env + eps)
    env = env / env_max
    bmode_db = 20 * np.log10(env + eps)

    meta = {
        "c": c,
        "f0": f0,
        "fs": fs,
        "Nelem": nelem,
        "pitch": pitch,
        "x_el": x_el,
        "x_img": x_img,
        "z_img": z_img,
        "SNR_dB": snr_db,
    }

    data = {
        "rf": rf,
        "t": t,
        "x_el": x_el,
        "x_img": x_img,
        "z_img": z_img,
        "bmode_dB": bmode_db,
        "env": env,
        "meta": meta,
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

    nx, nz = 128, 128
    x_img = np.linspace(-x_span / 2, x_span / 2, nx)
    z_img = np.linspace(z_min, z_max, nz)

    rf_analytic = hilbert(rf, axis=0)

    nelem = params.Nelem if nelem is None else nelem
    snr_db = params.SNR_dB if snr_db is None else snr_db
    pitch = params.pitch
    aperture = (nelem - 1) * pitch
    x_el = np.linspace(-aperture / 2, aperture / 2, nelem)
    bmode_mvdr = np.zeros((nz, nx), dtype=np.float32)
    y_align = np.zeros((nelem, nx, nz), dtype=np.complex64)

    z_max_toa = z_max / c
    r_max = np.sqrt((x_span / 2 + aperture / 2) ** 2 + z_max**2)
    t_max = z_max_toa + r_max / c + 2 / f0
    t = np.arange(0.0, t_max + 1.0 / fs, 1.0 / fs)

    print(f"MVDR Grid: {nx}x{nz} | Alignement des signaux...")

    for ix in range(nx):
        x0 = x_img[ix]
        for n in range(nelem):
            dx = x0 - x_el[n]
            rrx = np.sqrt(dx**2 + z_img**2)
            tau_tot = (z_img / c) + (rrx / c)

            val_real = np.interp(
                tau_tot, t, np.real(rf_analytic[:, n]), left=0.0, right=0.0
            )
            val_imag = np.interp(
                tau_tot, t, np.imag(rf_analytic[:, n]), left=0.0, right=0.0
            )
            y_align[n, ix, :] = val_real + 1j * val_imag

    print("Calcul des poids (Inv Covariance)...")

    a = np.ones((nelem, 1), dtype=np.complex64)
    eye_n = np.eye(nelem)
    rf_mvdr = np.zeros((nz, nx), dtype=np.float32)
    for iz in range(nz):
        for ix in range(nx):
            x_vec = y_align[:, ix, iz].reshape(-1, 1)
            r = x_vec @ x_vec.conj().T

            power = np.real(np.trace(r))
            if power < 1e-12:
                continue

            delta = regularization * power / nelem
            r_loaded = r + delta * eye_n

            try:
                num = np.linalg.solve(r_loaded, a)
                den = np.real(a.conj().T @ num).item()
                w = num / den
            except np.linalg.LinAlgError:
                bmode_mvdr[iz, ix] = 0.0
                continue

            pixel_val = (w.conj().T @ x_vec).item()
            rf_mvdr[iz, ix] = np.real(pixel_val)
            bmode_mvdr[iz, ix] = np.abs(pixel_val)

    val_max = np.max(bmode_mvdr)
    if val_max == 0:
        val_max = 1.0

    env = bmode_mvdr / val_max
    bmode_db = 20 * np.log10(env + 1e-12)

    meta = {
        "c": c,
        "f0": f0,
        "fs": fs,
        "Nelem": nelem,
        "pitch": pitch,
        "x_img": x_img,
        "z_img": z_img,
        "SNR_dB": snr_db,
    }

    data = {
        "rf": rf,
        "x_img": x_img,
        "z_img": z_img,
        "x_el": x_el,
        "bmode_dB": bmode_db,
        "env": env,
        "target_rf": rf_mvdr,
        "meta": meta,
    }
    return data
