from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import hann


def beamforming(rf: np.ndarray, nelem: int, snr_db: float) -> dict:
    c = 1540.0
    f0 = 5e6
    fs = 40e6

    x_span = 20e-3
    z_min = 10e-3
    z_max = 50e-3
    nx = 256
    nz = 256
    x_img = np.linspace(-x_span / 2, x_span / 2, nx)
    z_img = np.linspace(z_min, z_max, nz)

    bmode_lin = np.zeros((nz, nx), dtype=np.float32)
    y_align = np.zeros((nelem, nx, nz), dtype=np.float32)

    pitch = 0.15e-3
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

        tmp2 = apo_rx[:, None] * y_align[:, ix, :]
        y_sum = np.sum(tmp2, axis=0)
        bmode_lin[:, ix] = y_sum.astype(np.float32)

    pad_width = 64
    bmode_padded = np.pad(bmode_lin, ((pad_width, pad_width), (0, 0)), mode="constant")
    analytic_padded = hilbert(bmode_padded, axis=0)
    analytic = analytic_padded[pad_width:-pad_width, :]

    env = np.abs(analytic)
    env_max = np.max(env + eps)
    env = env / env_max
    bmode_dB = 20 * np.log10(env + eps)

    return {
        "rf": rf,
        "t": t,
        "x_el": x_el,
        "x_img": x_img,
        "z_img": z_img,
        "bmode_dB": bmode_dB,
        "env": env,
        "y_align": y_align,
        "meta": {
            "c": c,
            "f0": f0,
            "fs": fs,
            "Nelem": nelem,
            "pitch": pitch,
            "SNR_dB": snr_db,
        },
    }


def mvdr_beamforming(
    rf: np.ndarray,
    nelem: int,
    snr_db: float,
    regularization: float,
) -> dict:
    c = 1540.0
    f0 = 5e6
    fs = 40e6
    x_span = 20e-3
    z_min = 10e-3
    z_max = 50e-3

    nx = 64
    nz = 64

    x_img = np.linspace(-x_span / 2, x_span / 2, nx)
    z_img = np.linspace(z_min, z_max, nz)

    rf_analytic = hilbert(rf, axis=0)
    bmode_mvdr = np.zeros((nz, nx), dtype=np.float32)
    y_align = np.zeros((nelem, nx, nz), dtype=np.complex64)

    pitch = 0.15e-3
    aperture = (nelem - 1) * pitch
    x_el = np.linspace(-aperture / 2, aperture / 2, nelem)

    z_max_toa = z_max / c
    r_max = np.sqrt((x_span / 2 + aperture / 2) ** 2 + z_max**2)
    t_max = z_max_toa + r_max / c + 2 / f0
    t = np.arange(0.0, t_max + 1.0 / fs, 1.0 / fs)

    for ix in range(nx):
        x0 = x_img[ix]
        for n in range(nelem):
            dx = x0 - x_el[n]
            rrx = np.sqrt(dx**2 + z_img**2)
            tau_tot = (z_img / c) + (rrx / c)

            val_real = np.interp(tau_tot, t, np.real(rf_analytic[:, n]), left=0.0, right=0.0)
            val_imag = np.interp(tau_tot, t, np.imag(rf_analytic[:, n]), left=0.0, right=0.0)
            y_align[n, ix, :] = val_real + 1j * val_imag

    a = np.ones((nelem, 1), dtype=np.complex64)

    for iz in range(nz):
        for ix in range(nx):
            x_vec = y_align[:, ix, iz].reshape(-1, 1)
            r_mat = x_vec @ x_vec.conj().T
            power = np.trace(r_mat).real
            if power < 1e-12:
                bmode_mvdr[iz, ix] = 0.0
                continue

            delta = regularization * power / nelem
            r_loaded = r_mat + delta * np.eye(nelem)
            try:
                num = np.linalg.solve(r_loaded, a)
            except np.linalg.LinAlgError:
                num = np.linalg.pinv(r_loaded) @ a
            den = a.conj().T @ num
            w = num / den
            pixel_val = w.conj().T @ x_vec
            bmode_mvdr[iz, ix] = float(np.abs(pixel_val).item())

    val_max = np.max(bmode_mvdr)
    if val_max == 0:
        val_max = 1.0
    bmode_dB = 20 * np.log10(bmode_mvdr / val_max + 1e-12)

    return {
        "rf": rf,
        "x_img": x_img,
        "z_img": z_img,
        "bmode_dB": bmode_dB,
        "meta": {
            "c": c,
            "f0": f0,
            "fs": fs,
            "Nelem": nelem,
            "pitch": pitch,
            "SNR_dB": snr_db,
            "regularization": regularization,
        },
    }
