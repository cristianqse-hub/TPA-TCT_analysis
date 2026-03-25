import numpy as np

from utils_lib import getVals, wu_rootfile, reshape_paramReps
from pathlib import Path

def analyze_wfsraw(
    root_path: str,
    do_aTleft: bool = False,
    do_aTRight: bool = False,
    tleft_frac: float = 0.10,   # <-- NUEVO: 10% por defecto
    tright_frac: float = 0.001,  # opcional si querés lo mismo a la derecha
):
    """
    Compute basic signal properties from Raw Signals.
    TLeft ahora se calcula como el cruce (en el segmento izquierdo) del umbral
    BL + frac*(max-BL). Por defecto frac=0.10 (10%).
    """
    vals = getVals(root_path, ["Raw:WFsRaw", "Raw:aTBL", "Raw:t"])
    wfs_raw = np.asarray(vals["Raw:WFsRaw"])
    aTBL = float(vals["Raw:aTBL"])
    t = np.asarray(vals["Raw:t"])

    n_events = wfs_raw.shape[0]

    maxV = np.max(wfs_raw, axis=1)
    minV = np.min(wfs_raw, axis=1)

    bl_mask = t <= aTBL
    if np.any(bl_mask):
        BLLevel = np.mean(wfs_raw[:, bl_mask], axis=1)
        noise = np.std(wfs_raw[:, bl_mask], axis=1)
    else:
        BLLevel = np.full(n_events, np.nan)
        noise = np.full(n_events, np.nan)

    TLeft = np.full(n_events, np.nan)
    TRight = np.full(n_events, np.nan)

    # validación simple
    if not (0.0 < tleft_frac < 1.0):
        raise ValueError("tleft_frac debe estar entre 0 y 1 (ej: 0.10 para 10%).")

    for idx in range(n_events):
        wf = wfs_raw[idx]
        max_idx = int(np.argmax(wf))

        bl = BLLevel[idx]
        if not np.isfinite(bl):
            continue

        amp = maxV[idx] - bl
        if not np.isfinite(amp) or amp <= 0:
            continue

        # --- TLeft: cruce del umbral en el segmento izquierdo ---
        thr_left = bl + tleft_frac * amp

        left = wf[:max_idx + 1]          # incluye el pico
        left_t = t[:max_idx + 1]

        # buscamos el último punto <= thr antes del pico (equiv: primer <= thr al ir hacia atrás)
        rev = left[::-1]
        rev_idx = np.where(rev <= thr_left)[0]

        if rev_idx.size > 0:
            j = max_idx - rev_idx[0]     # índice en el arreglo "left" (y en wf/t)

            # si j == max_idx, justo en el pico ya estaba bajo el umbral (raro, pero posible)
            # si j < max_idx, interpolamos entre j y j+1 para el cruce
            if j < max_idx:
                y0, y1 = wf[j], wf[j + 1]
                x0, x1 = t[j], t[j + 1]

                # interpolación lineal si hay pendiente
                if y1 != y0:
                    alpha = (thr_left - y0) / (y1 - y0)
                    TLeft[idx] = x0 + alpha * (x1 - x0)
                else:
                    TLeft[idx] = x0
            else:
                TLeft[idx] = t[j]

        # --- TRight (opcional): igual idea hacia la derecha ---
        if tright_frac is not None:
            if not (0.0 < tright_frac < 1.0):
                raise ValueError("tright_frac debe estar entre 0 y 1.")
            thr_right = bl + tright_frac * amp

            right = wf[max_idx:]         # desde el pico
            right_t = t[max_idx:]

            # primer punto <= thr al avanzar
            ridx = np.where(right <= thr_right)[0]
            if ridx.size > 0:
                k = ridx[0]
                if k > 0:
                    y0, y1 = right[k - 1], right[k]
                    x0, x1 = right_t[k - 1], right_t[k]
                    if y1 != y0:
                        alpha = (thr_right - y0) / (y1 - y0)
                        TRight[idx] = x0 + alpha * (x1 - x0)
                    else:
                        TRight[idx] = x1
                else:
                    TRight[idx] = right_t[k]
        else:
            # comportamiento viejo para TRight: cruce por 0
            right_indices = np.where(wf[max_idx:] <= 0)[0]
            if right_indices.size > 0:
                TRight[idx] = t[max_idx + right_indices[0]]

    features_names = ["maxV", "minV", "BLLevel", "noise", "TLeft", "TRight"]
    features_values = [maxV, minV, BLLevel, noise, TLeft, TRight]

    if do_aTleft:
        features_names.append("aTLeft")
        features_values.append(float(np.nanmean(TLeft)))

    if do_aTRight:
        features_names.append("aTRight")
        features_values.append(float(np.nanmean(TRight)))

    wu_rootfile(root_path, features_names, features_values, "Raw")
    return None

def get_signalsROI(
    root_path: str,
    interp_nsamples: int = 5,
    do_aTleft: bool = False,
    do_aTRight: bool = False,
    mask_ignoreLeft: bool = False,
    mask_ignoreRight: bool = False,
    # --- NUEVO: umbrales (caída desde el pico) ---
    tleft_drop: float = 0.95,   # 10% de caída => umbral al 90% del pico
    tright_drop: float = 0.99,  # idem derecha
    use_crossing: bool = True,  # True: cruce + interpolación; False: primer sample <= thr
):
    """
    Generate ROI from signals with mask.

    - Baseline se resta (BL -> 0).
    - TLeft_sig/TRight_sig se calculan en ROI usando umbral fraccional:
        thr = (1 - drop) * max(wf)
      (pulsos positivos).
    - Si el cruce no se encuentra, se deja aTLeft/aTRight como fallback.
    """
    vals = getVals(
        root_path,
        ["Raw:WFsRaw", "Raw:aTLeft", "Raw:aTRight", "Raw:t", "Raw:BLLevel"],
    )
    wfs_raw = np.asarray(vals["Raw:WFsRaw"])
    bl_level = np.asarray(vals["Raw:BLLevel"])
    t_raw = np.asarray(vals["Raw:t"])

    if bl_level.ndim == 0:
        bl_level = np.full(wfs_raw.shape[0], float(bl_level))

    # baseline a 0
    wfs_raw = wfs_raw - bl_level[:, None]

    n_events = wfs_raw.shape[0]

    aTLeft = float(vals["Raw:aTLeft"])
    aTRight = float(vals["Raw:aTRight"])

    # ROI definido por promedios como en tu código
    TLeft = np.full(n_events, aTLeft)
    TRight = np.full(n_events, aTRight)

    finite = np.isfinite(TLeft) & np.isfinite(TRight)
    if not np.any(finite):
        raise ValueError("No hay valores finitos en aTLeft/aTRight")

    t_min = float(np.nanmin(TLeft[finite]))
    t_max = float(np.nanmax(TRight[finite]))
    iTLeft = int(np.searchsorted(t_raw, t_min, side="left"))
    iTRight = int(np.searchsorted(t_raw, t_max, side="right"))

    if iTRight <= iTLeft:
        if np.isclose(t_min, t_max):
            iTRight = min(iTLeft + 1, t_raw.size)
        else:
            raise ValueError(
                f"ROI vacío: iTLeft={iTLeft}, iTRight={iTRight}, "
                f"t_min={t_min}, t_max={t_max}"
            )

    wfs_roi = wfs_raw[:, iTLeft:iTRight]
    t_roi = t_raw[iTLeft:iTRight]

    if t_roi.size == 0:
        raise ValueError(
            f"ROI sin muestras: iTLeft={iTLeft}, iTRight={iTRight}, "
            f"t_raw.size={t_raw.size}"
        )

    t = np.linspace(t_roi[0], t_roi[-1], max(2, len(t_roi) * interp_nsamples))
    signals = np.vstack([np.interp(t, t_roi, row) for row in wfs_roi])

    maxV = np.max(signals, axis=1)
    minV = np.min(signals, axis=1)
    BLLevel = np.zeros(n_events)

    # FIX: fallback correcto (antes tenías aTRight en TLeft_sig)
    TLeft_sig = np.full(n_events, aTLeft)
    TRight_sig = np.full(n_events, aTRight)

    if not (0.0 < tleft_drop < 1.0):
        raise ValueError("tleft_drop debe estar entre 0 y 1 (ej: 0.10).")
    if not (0.0 < tright_drop < 1.0):
        raise ValueError("tright_drop debe estar entre 0 y 1 (ej: 0.10).")

    for idx in range(n_events):
        wf = signals[idx]
        max_idx = int(np.argmax(wf))
        peak = wf[max_idx]

        if not np.isfinite(peak) or peak <= 0:
            continue

        thr_left = (1.0 - tleft_drop) * peak
        thr_right = (1.0 - tright_drop) * peak

        # ---- TLeft: en el segmento izquierdo del máximo ----
        left = wf[:max_idx + 1]
        left_t = t[:max_idx + 1]

        if use_crossing:
            # cruce en la subida: left[i] <= thr y left[i+1] > thr
            cross = np.where((left[:-1] <= thr_left) & (left[1:] > thr_left))[0]
            if cross.size > 0:
                i = cross[-1]  # el más cercano al pico
                y0, y1 = left[i], left[i + 1]
                x0, x1 = left_t[i], left_t[i + 1]
                if y1 != y0:
                    alpha = (thr_left - y0) / (y1 - y0)
                    TLeft_sig[idx] = x0 + alpha * (x1 - x0)
                else:
                    TLeft_sig[idx] = x0
        else:
            # primer sample bajando desde el pico que cae <= thr_left
            rev = left[::-1]
            ridx = np.where(rev <= thr_left)[0]
            if ridx.size > 0:
                j = max_idx - ridx[0]
                TLeft_sig[idx] = t[j]

        # ---- TRight: en el segmento derecho del máximo ----
        right = wf[max_idx:]
        right_t = t[max_idx:]

        if use_crossing:
            # cruce en la bajada: right[i] > thr y right[i+1] <= thr
            cross_r = np.where((right[:-1] > thr_right) & (right[1:] <= thr_right))[0]
            if cross_r.size > 0:
                i = cross_r[0]  # el primero después del pico
                y0, y1 = right[i], right[i + 1]
                x0, x1 = right_t[i], right_t[i + 1]
                if y1 != y0:
                    alpha = (thr_right - y0) / (y1 - y0)
                    TRight_sig[idx] = x0 + alpha * (x1 - x0)
                else:
                    TRight_sig[idx] = x1
        else:
            ridx = np.where(right <= thr_right)[0]
            if ridx.size > 0:
                TRight_sig[idx] = right_t[ridx[0]]

    mask = np.ones_like(signals, dtype=bool)
    if not mask_ignoreLeft:
        mask &= t[None, :] >= TLeft_sig[:, None]
    if not mask_ignoreRight:
        mask &= t[None, :] <= TRight_sig[:, None]

    signals_masked = np.where(mask, signals, np.nan)

    features_names = [
        "maxV", "minV", "BLLevel", "TLeft", "TRight",
        "signals", "signals_masked", "t", "mask",
    ]
    features_values = [
        maxV, minV, BLLevel, TLeft_sig, TRight_sig,
        signals, signals_masked, t, mask.astype(np.int8),
    ]

    if do_aTleft:
        features_names.append("aTLeft")
        features_values.append(float(np.nanmean(TLeft_sig)))
    else:
        try:
            raw_vals = getVals(root_path, ["Raw:aTLeft"])
            features_names.append("aTLeft")
            features_values.append(float(raw_vals["Raw:aTLeft"]))
        except Exception:
            pass

    if do_aTRight:
        features_names.append("aTRight")
        features_values.append(float(np.nanmean(TRight_sig)))
    else:
        try:
            raw_vals = getVals(root_path, ["Raw:aTRight"])
            features_names.append("aTRight")
            features_values.append(float(raw_vals["Raw:aTRight"]))
        except Exception:
            pass

    wu_rootfile(root_path, features_names, features_values, "Signal")
    return None


def get_signalsROIddd(
    root_path: str,
    interp_nsamples=5,
    do_aTleft: bool = False,
    do_aTRight: bool = False,
    mask_ignoreLeft: bool = False,
    mask_ignoreRight: bool = False,
):
    """
    Generate ROI from signals with mask.
    """
    vals = getVals(
        root_path,
        ["Raw:WFsRaw", "Raw:aTLeft", "Raw:aTRight", "Raw:t", "Raw:BLLevel"],
    )
    wfs_raw = np.asarray(vals["Raw:WFsRaw"])
    bl_level = np.asarray(vals["Raw:BLLevel"])
    t_raw = np.asarray(vals["Raw:t"])

    if bl_level.ndim == 0:
        bl_level = np.full(wfs_raw.shape[0], float(bl_level))

    wfs_raw = wfs_raw - bl_level[:, None]

    n_events = wfs_raw.shape[0]
    n_samples = wfs_raw.shape[1]

    aTLeft = float(vals["Raw:aTLeft"])
    aTRight = float(vals["Raw:aTRight"])

    TLeft = np.full(n_events, aTLeft)
    TRight = np.full(n_events, aTRight)

    finite = np.isfinite(TLeft) & np.isfinite(TRight)
    if not np.any(finite):
        raise ValueError("No hay valores finitos en aTLeft/aTRight")

    t_min = float(np.nanmin(TLeft[finite]))
    t_max = float(np.nanmax(TRight[finite]))
    iTLeft = int(np.searchsorted(t_raw, t_min, side="left"))
    iTRight = int(np.searchsorted(t_raw, t_max, side="right"))

    if iTRight <= iTLeft:
        if np.isclose(t_min, t_max):
            iTRight = min(iTLeft + 1, t_raw.size)
        else:
            raise ValueError(
                f"ROI vacío: iTLeft={iTLeft}, iTRight={iTRight}, "
                f"t_min={t_min}, t_max={t_max}"
            )

    wfs_roi = wfs_raw[:, iTLeft:iTRight]
    t_roi = t_raw[iTLeft:iTRight]

    if t_roi.size == 0:
        raise ValueError(
            f"ROI sin muestras: iTLeft={iTLeft}, iTRight={iTRight}, "
            f"t_raw.size={t_raw.size}"
        )

    t = np.linspace(t_roi[0], t_roi[-1], max(2, len(t_roi) * interp_nsamples))
    signals = np.vstack([np.interp(t, t_roi, row) for row in wfs_roi])

    maxV = np.max(signals, axis=1)
    minV = np.min(signals, axis=1)
    BLLevel = np.zeros(n_events)

    TLeft_sig = np.full(n_events, aTRight)
    TRight_sig = np.full(n_events, aTRight)

    for idx in range(n_events):
        wf = signals[idx]
        max_idx = int(np.argmax(wf))
        left_segment = wf[:max_idx + 1]
        left_rev_indices = np.where(left_segment[::-1] <= 0)[0]
        if left_rev_indices.size > 0:
            left_idx = max_idx - left_rev_indices[0]
            TLeft_sig[idx] = t[left_idx]

        right_indices = np.where(wf[max_idx:] <= 0)[0]
        if right_indices.size > 0:
            TRight_sig[idx] = t[max_idx + right_indices[0]]

    mask = np.ones_like(signals, dtype=bool)
    if not mask_ignoreLeft:
        mask &= t[None, :] >= TLeft_sig[:, None]
    if not mask_ignoreRight:
        mask &= t[None, :] <= TRight_sig[:, None]

    signals_masked = np.where(mask, signals, np.nan)

    features_names = [
        "maxV",
        "minV",
        "BLLevel",
        "TLeft",
        "TRight",
        "signals",
        "signals_masked",
        "t",
        "mask",
    ]
    features_values = [
        maxV,
        minV,
        BLLevel,
        TLeft_sig,
        TRight_sig,
        signals,
        signals_masked,
        t,
        mask.astype(np.int8),
    ]

    if do_aTleft:
        features_names.append("aTLeft")
        features_values.append(float(np.mean(TLeft_sig)))
    else:
        try:
            raw_vals = getVals(root_path, ["Raw:aTLeft"])
            features_names.append("aTLeft")
            features_values.append(float(raw_vals["Raw:aTLeft"]))
        except Exception:
            pass

    if do_aTRight:
        features_names.append("aTRight")
        features_values.append(float(np.mean(TRight_sig)))
    else:
        try:
            raw_vals = getVals(root_path, ["Raw:aTRight"])
            features_names.append("aTRight")
            features_values.append(float(raw_vals["Raw:aTRight"]))
        except Exception:
            pass

    wu_rootfile(root_path, features_names, features_values, "Signal")

    return None

def analyze_signalsROI(
    root_path: str,
    signals_spec: str = "Signal:signals",
    output_tree: str = "Signal",
    thresholds=None,
):
    """
    Compute peak time, collection times, and SNR from ROI signals.
    """
    if thresholds is None:
        thresholds = [0, 5, 10, 25, 50]

    if ":" in signals_spec:
        tree_name, _ = signals_spec.split(":", 1)
        t_spec = f"{tree_name}:t"
    else:
        t_spec = "Signal:t"

    vals = getVals(root_path, [signals_spec, t_spec, "Raw:noise"])
    signals = np.asarray(vals[signals_spec])
    t = np.asarray(vals[t_spec])
    noise = np.asarray(vals["Raw:noise"])

    n_events = signals.shape[0]
    n_thresholds = len(thresholds)

    if noise.ndim == 0:
        noise = np.full(n_events, float(noise))

    peakRime = np.full(n_events, np.nan)
    tColl = np.full((n_thresholds, n_events), np.nan)
    SNR = np.full(n_events, np.nan)

    for idx in range(n_events):
        sig = signals[idx]
        if sig.size == 0 or np.all(np.isnan(sig)):
            continue
        max_idx = int(np.nanargmax(sig))
        max_val = sig[max_idx]
        if max_idx < t.size:
            peakRime[idx] = t[max_idx]

        if noise[idx] != 0:
            SNR[idx] = max_val / noise[idx]

        for jdx, thr in enumerate(thresholds):
            level = max_val * (float(thr) / 100.0)
            left_segment = sig[: max_idx + 1]
            left_rev_indices = np.where(left_segment[::-1] <= level)[0]
            right_indices = np.where(sig[max_idx:] <= level)[0]
            if left_rev_indices.size == 0 or right_indices.size == 0:
                continue
            left_idx = max_idx - left_rev_indices[0]
            right_idx = max_idx + right_indices[0]
            if right_idx < t.size and left_idx < t.size:
                tColl[jdx, idx] = t[right_idx] - t[left_idx]

    wu_rootfile(
        root_path,
        ["peakRime", "tColl", "SNR"],
        [peakRime, tColl, SNR],
        output_tree,
    )

    return None

def integrate_charge(
    root_path: str,
    t_spec: str,
    signals_spec: str,
    saveto: str,
    unitfactor=1,
):
    """
    Integrate signals with trapezoidal rule and store charge vector.
    """
    vals = getVals(root_path, [t_spec, signals_spec])
    t = np.asarray(vals[t_spec])
    signals = np.asarray(vals[signals_spec])

    signals = np.nan_to_num(signals, nan=0.0)

    charge = np.trapz(signals, x=t, axis=1) * float(unitfactor)

    if ":" not in saveto:
        raise ValueError("saveto debe tener formato 'tree:param'")
    tree_name, param = saveto.split(":", 1)
    tree_name = tree_name.strip()
    param = param.strip()
    if not tree_name or not param:
        raise ValueError("saveto debe tener formato 'tree:param'")

    wu_rootfile(root_path, [param], [charge], tree_name)

    return charge

def correct_Signals(
    root_path: str,
    mode: str = "COR",  # Mode COR - TPA
    tpa_cor_factor = 2.0, 
    spa_WFsamplesNumber=10,
    smoothing=False,
):
    vals = getVals(root_path, ["Signal:signals", "Signal:mask", "Raw:LP"])
    signals = np.asarray(vals["Signal:signals"])
    mask = np.asarray(vals["Signal:mask"])
    LP = np.asarray(vals["Raw:LP"])

    n_events = signals.shape[0]
    n_samples = signals.shape[1]

    if mode == "COR":  # RAW TPA = S / LP^2 -> No spa substraction
        signals_COR = signals / (LP[:, np.newaxis] ** tpa_cor_factor)
        signals_COR_masked = np.where(mask, signals_COR, np.nan)

        wu_rootfile(
            root_path,
            ["signals_COR", "signals_COR_masked"],
            [signals_COR, signals_COR_masked],
            "Signal",
        )
    if mode == "TPA":
        temp_signals = signals / LP[:, np.newaxis]
        amplitudes = np.nanmax(temp_signals, axis=1)
        order = np.argsort(amplitudes)
        n_spa = max(1, min(int(spa_WFsamplesNumber), n_events))
        spa_aver_wf = np.nanmean(temp_signals[order[:n_spa]], axis=0)
        if smoothing:
            spa_aver_wf = np.convolve(spa_aver_wf, np.ones(3) / 3, mode="same")

        tpa_uncorrected = temp_signals - spa_aver_wf[None, :]
        signals_TPA = tpa_uncorrected / LP[:, np.newaxis]
        signals_TPA_masked = np.where(mask, signals_TPA, np.nan)

        wu_rootfile(
            root_path,
            ["signals_TPA", "signals_TPA_masked"],
            [signals_TPA, signals_TPA_masked],
            "Signal",
        )




def zscan_profileAnalisis(
    root_file: str,
    charge_profile: str,
    mode: str = "Partial",
    partial_threshold: float = 1.0,
    show_plot: bool = True,
):
    """
    Zscan charge profile analysis.
    modes:
        "Partial" -> Single fit for rising edge.
    """
    from scipy.optimize import curve_fit
    import plotly.graph_objects as go
    from scipy.interpolate import interp1d

    def func_ChargePartial(x, a, z0, Rl, SPA):
        return a * ((np.arctan((x-z0)/Rl) + np.pi/2) / np.pi) + SPA
    
    def func_ChargeAsymetric(x, a, z0, z1, Rl0, Rl1, SPA):
        return a * ((np.arctan((x-z0)/Rl0) + np.pi/2) / np.pi) * ((np.arctan((z1-x)/Rl1) + np.pi/2) / np.pi)  + SPA

    vals = getVals(root_file, ["Raw:z_R", f"{charge_profile}_R"])
    z = np.asarray(vals["Raw:z_R"])
    charge = np.asarray(vals[f"{charge_profile}_R"])

    reps = charge.shape[0]
    features_names = []
    features_pars = []

    

    ### MODES

    if mode == "Partial":
        spa = np.full(reps, np.nan)
        z0 = np.full(reps, np.nan)
        Rl = np.full(reps, np.nan)

        fig = go.Figure() if show_plot else None

        for i in range(reps):
            _charge = charge[i, :].astype(float, copy=True)
            max_val = np.nanmax(_charge)
            if max_val != 0:
                _charge = _charge / max_val

            idx = np.argmax(_charge >= partial_threshold)
            if _charge[idx] < partial_threshold:
                idx = _charge.size

            _z = z[i, :idx]
            _charge = _charge[:idx]

            #
            idxP0 = np.argmax(_charge >= 0.5)
            if _charge[idxP0] < partial_threshold:
                idxP0 = _charge.size

            try:
                p, _ = curve_fit(
                    func_ChargePartial,
                    _z,
                    _charge,
                    p0=[1, z[i, idxP0], 1.5, 0],
                    bounds=(
                        [-np.inf, -np.inf, -np.inf, -np.inf],  # límites inferiores
                        [ np.inf,  np.inf, np.inf, np.inf]  # superiores
                    )
                )

                z0[i] = p[1]
                Rl[i] = p[2]
                spa[i] = p[3] * max_val

                if show_plot:
                    z_fit = np.linspace(_z.min(), _z.max(), 100)
                    charge_fit = func_ChargePartial(z_fit, *p)
                    fig.add_trace(
                        go.Scatter(x=_z, y=_charge, mode="markers", showlegend=False)
                    )
                    fig.add_trace(
                        go.Scatter(x=z_fit, y=charge_fit, mode="lines", showlegend=False)
                    )

            except Exception:
                print(f"Fit for file {root_file} rep={i} has failed")

        features_names += ["z0", "Rl", "spa"]
        features_pars += [z0, Rl, spa]

        if show_plot:
            fig.update_layout(
                title=f"Zscan - profile fit ({charge_profile})",
                xaxis_title="z",
                yaxis_title="charge (norm)",
            )
            fig.show(renderer="png")

    if mode == "Asymetric":
        spa = np.full(reps, np.nan)
        z0 = np.full(reps, np.nan)
        Rl = np.full(reps, np.nan)

        fig = go.Figure() if show_plot else None

        for i in range(reps):
            _charge = charge[i, :].astype(float, copy=True)
            max_val = np.nanmax(_charge)
            if max_val != 0:
                _charge = _charge / max_val

            _z = z[i, :]
            _charge = _charge[:]

            try:
                p, _ = curve_fit(
                    func_ChargeAsymetric,
                    _z,
                    _charge,
                    p0=[1, 20, 40, 1.5, 3.0, 0]
                )

                z0[i] = p[1]
                Rl[i] = p[3]
                spa[i] = p[5] * max_val

                if show_plot:
                    z_fit = np.linspace(_z.min(), _z.max(), 100)
                    charge_fit = func_ChargeAsymetric(z_fit, *p)
                    fig.add_trace(
                        go.Scatter(x=_z, y=_charge, mode="markers", showlegend=False)
                    )
                    fig.add_trace(
                        go.Scatter(x=z_fit, y=charge_fit, mode="lines", showlegend=False)
                    )

            except Exception:
                print(f"Fit for file {root_file} rep={i} has failed")

        features_names += ["z0", "Rl", "spa"]
        features_pars += [z0, Rl, spa]

        if show_plot:
            fig.update_layout(
                title=f"Zscan - profile fit ({charge_profile})",
                xaxis_title="z",
                yaxis_title="charge (norm)",
            )
            fig.show(renderer="png")

    ### BASE parameters
    FWHM = np.full(reps, np.nan)
    for i in range(reps):
        _charge = charge[i, :].astype(float, copy=True) 
        _z = z[i, :].astype(float, copy=True)
        # Interpolate
        z_int = np.linspace(_z.min(), _z.max(), int(_z.size) * 10)
        f = interp1d(_z, _charge, kind="linear", bounds_error=False, fill_value=np.nan)
        charge_int = f(z_int)
        charge_int /= np.nanmax(charge_int)
        FWHM[i] = len(charge_int[ (charge_int >= 0.5) ]) * (z_int[1] - z_int[0])

    width = np.full(reps, np.nan)
    for i in range(reps):
        width[i] = 2*np.sqrt(FWHM[i]**2/4-(Rl[i])**2)


    features_names += ["FWHM", "width"]
    features_pars += [FWHM, width]

    if show_plot:
        print(Path(root_file).stem)
        for i, name in enumerate(features_names):
            print(f"Param {name} = {features_pars[i]}" )

    # all charge vector in one
    z_cor = z - z0[:, np.newaxis]
    z_vec = z_cor.ravel()
    charge_vec = charge.ravel()

    zcharge = np.zeros([2, len(z_vec)])
    # Short both by z
    idx = np.argsort(z_vec)
    zcharge[0, :] = z_vec[idx] 
    zcharge[1, :] = charge_vec[idx] 

    features_names += ["zcharge"]
    features_pars += [zcharge]

    if features_names:
        wu_rootfile(root_file, features_names, features_pars, "zscan")



def analyce_signalsWPC(root_path, t_pc, normPar, signals_spec, savetoTree="Signal"):
    """
    For each event, compute WPC as the first value of the signal (normalised by a
    scalar weight) whose compacted time (0, dt, 2dt, ...) is >= t_pc.
    """

    if ":" in signals_spec:
        tree_name, _ = signals_spec.split(":", 1)
        t_spec = f"{tree_name}:t"
    else:
        t_spec = "Signal:t"

    vals    = getVals(root_path, [normPar, signals_spec, t_spec])
    weight  = np.asarray(vals[normPar])        # scalar normalisation per event
    signals = np.asarray(vals[signals_spec])   # shape: (events, samples)
    t       = np.asarray(vals[t_spec])

    WPC = []
    dt  = t[1] - t[0]

    # First time sample satisfying t >= t_pc
    real_t_pc = t[np.searchsorted(t, t_pc, side="left")]

    n_events = signals.shape[0]
    for i in range(n_events):
        # Remove NaNs from signal, keep scalar normalisation
        weightedSignal = signals[i, ~np.isnan(signals[i])] / weight[i]

        # Time starts with unmasked values (this is really not needed, we could use t)
        unmaskedTime = np.arange(len(weightedSignal)) * dt

        # First value with time >= t_pc
        WPC.append(weightedSignal[unmaskedTime >= t_pc][0])

    wu_rootfile(root_path, ["t_pc", "WPC"], [real_t_pc, np.array(WPC)], savetoTree)

