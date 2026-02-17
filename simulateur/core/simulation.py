import numpy as np


def simulate_us_scene(
    params,
    scene,
    rng,
    snr_db=None,
    nelem=None,
):
    snr_db = params.SNR_dB if snr_db is None else snr_db
    nelem = params.Nelem if nelem is None else nelem

    x_span = params.x_span
    z_min = params.z_min
    z_max = params.z_max
    c = params.c
    p = params.p
    f0 = params.f0
    fs = params.fs

    pitch = params.pitch
    aperture = (nelem - 1) * pitch
    x_el = np.linspace(-aperture / 2, aperture / 2, nelem)

    nCycles = params.nCycles
    pulseT = nCycles / f0
    t_pulse = np.arange(-pulseT, pulseT + 1.0 / fs, 1.0 / fs)
    sigma_t = pulseT / 2.355
    pulse = np.cos(2 * np.pi * f0 * t_pulse) * np.exp(
        -(t_pulse ** 2) / (2 * sigma_t ** 2)
    )

    z_max_toa = z_max / c
    r_max = np.sqrt((x_span / 2 + aperture / 2) ** 2 + z_max ** 2)
    t_max = z_max_toa + r_max / c + 2 / f0
    t = np.arange(0.0, t_max + 1.0 / fs, 1.0 / fs)
    Nt = t.size

    bright_points = np.array(scene.points, dtype=np.float32)
    if bright_points.size == 0:
        bright_points = bright_points.reshape((0, 3))

    xs = bright_points[:, 0]
    zs = bright_points[:, 1]
    as_ = bright_points[:, 2]

    N_scatt = as_.size

    if scene.layers:
        couches_data = []
        for layer in scene.layers:
            couches_data.append([layer.z_min, layer.z_max, layer.c, layer.rho])
        couches = np.array(couches_data, dtype=np.float32)
    else:
        couches = np.empty((0, 4), dtype=np.float32)

    sig_c = np.zeros(Nt, dtype=np.float32)
    coef_ref = None
    if couches.shape[0] > 0:
        impedence_i = couches[:, 2] * couches[:, 3]
        impedence = p * c
        coef_ref = (impedence_i - impedence) / (impedence_i + impedence)
        coef_ref = np.where(coef_ref < 0, np.abs(coef_ref), coef_ref)

        for n in range(couches.shape[0]):
            d1 = couches[n, 0]
            e = couches[n, 1] - couches[n, 0]
            t1 = 2 * d1 / c
            t2 = t1 + e * 2 / couches[n, 2]
            att1 = 1 / np.maximum((d1 * 1e3) ** 2, 1e-3) * coef_ref[n]
            att2 = (
                1
                / np.maximum(((d1 + e) * 1e3) ** 2, 1e-3)
                * coef_ref[n]
                * (1 - coef_ref[n]) ** 2
            )
            sig_c += np.float32(
                att1 * np.interp(t, t_pulse + t1, pulse, left=0.0, right=0.0)
            )
            sig_c += np.float32(
                att2 * np.interp(t, t_pulse + t2, pulse, left=0.0, right=0.0)
            )

    rf = np.zeros((Nt, nelem), dtype=np.float32)
    for n in range(nelem):
        dx_n = xs - x_el[n]
        Rrx = np.sqrt(dx_n ** 2 + zs ** 2)

        t_tx = zs / c
        t_rx = Rrx / c
        att_layer = np.ones(N_scatt, dtype=np.float32)

        for i_layer in range(couches.shape[0]):
            z_min_layer = couches[i_layer, 0]
            z_max_layer = couches[i_layer, 1]
            v_layer = couches[i_layer, 2]

            thickness_crossed = np.maximum(
                0, np.minimum(zs, z_max_layer) - z_min_layer
            )

            if np.any(thickness_crossed > 0):
                delta_tx = thickness_crossed * (1.0 / v_layer - 1.0 / c)
                t_tx = t_tx + delta_tx

                ratio = np.divide(
                    thickness_crossed,
                    zs,
                    out=np.zeros_like(thickness_crossed),
                    where=zs > 0,
                )
                dist_oblique_layer = Rrx * ratio
                delta_rx = dist_oblique_layer * (1.0 / v_layer - 1.0 / c)
                t_rx = t_rx + delta_rx

                full_cross = thickness_crossed >= (z_max_layer - z_min_layer)
                att_layer = np.where(
                    full_cross,
                    coef_ref[i_layer] ** 2 * (1 - coef_ref[i_layer]) ** 2,
                    coef_ref[i_layer] * (1 - coef_ref[i_layer]),
                )

        tau = t_tx + t_rx

        if couches.shape[0] > 0:
            att = (1.0 / np.maximum((Rrx * 1e3) ** 2, 1e-3)) * att_layer
        else:
            att = 1.0 / np.maximum((Rrx * 1e3) ** 2, 1e-3)

        sig_n = np.zeros(Nt, dtype=np.float32)
        for k in range(N_scatt):
            tk = tau[k]
            ak = as_[k] * att[k]
            sig_n += np.float32(
                ak * np.interp(t, t_pulse + tk, pulse, left=0.0, right=0.0)
            )
        rf[:, n] = sig_n + sig_c

    eps = np.finfo(np.float32).eps
    signal_pow = np.mean(rf.astype(np.float64) ** 2 + eps)
    noise_pow = signal_pow / (10 ** (snr_db / 10))
    rf = rf + np.sqrt(noise_pow).astype(np.float32) * rng.standard_normal(
        rf.shape
    ).astype(np.float32)

    return rf
