import numpy as np

from utils_lib import getVals, wu_rootfile
from pathlib import Path

def analyze_wfsraw(
    root_path: str,
    do_aTleft: bool = False,
    do_aTRight: bool = False,
):
    """
    Compute basic signal properties from Raw Signals.
    """
    vals = getVals(root_path, ["Raw:WFsRaw", "Raw:aTBL", "Raw:t"])
    wfs_raw = np.asarray(vals["Raw:WFsRaw"])
    aTBL = float(vals["Raw:aTBL"])
    t = np.asarray(vals["Raw:t"])

    n_events = wfs_raw.shape[0]
    n_samples = wfs_raw.shape[1]

    maxV = np.max(wfs_raw, axis=1)
    minV = np.min(wfs_raw, axis=1)

    bl_mask = t <= aTBL
    if np.any(bl_mask):
        BLLevel = np.mean(wfs_raw[:, bl_mask], axis=1)
        noise = np.std(wfs_raw[:, bl_mask], axis=1)
    else:
        BLLevel = np.full(n_events, np.nan)

    TLeft = np.full(n_events, np.nan)
    TRight = np.full(n_events, np.nan)

    for idx in range(n_events):
        wf = wfs_raw[idx]
        max_idx = int(np.argmax(wf))
        left_segment = wf[:max_idx + 1]
        left_rev_indices = np.where(left_segment[::-1] <= 0)[0]
        if left_rev_indices.size > 0:
            left_idx = max_idx - left_rev_indices[0]
            TLeft[idx] = t[left_idx]

        right_indices = np.where(wf[max_idx:] <= 0)[0]
        if right_indices.size > 0:
            TRight[idx] = t[max_idx + right_indices[0]]

    features_names = [
        "maxV",
        "minV",
        "BLLevel",
        "noise",
        "TLeft",
        "TRight",
    ]
    features_values = [
        maxV,
        minV,
        BLLevel,
        noise,
        TLeft,
        TRight,
    ]

    if do_aTleft:
        features_names.append("aTLeft")
        features_values.append(float(np.mean(TLeft)))

    if do_aTRight:
        features_names.append("aTRight")
        features_values.append(float(np.mean(TRight)))

    wu_rootfile(root_path, features_names, features_values, "Raw")

    return None


def get_signalsROI(
    root_path: str,
    interp_nsamples=5,
    do_aTleft: bool = False,
    do_aTRight: bool = False,
    mask_ignoreLeft: bool = True,
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
):
    vals = getVals(root_path, ["Signal:signals", "Signal:mask", "Raw:LP"])
    signals = np.asarray(vals["Signal:signals"])
    mask = np.asarray(vals["Signal:mask"])
    LP = np.asarray(vals["Raw:LP"])

    n_events  = signals.shape[0]
    n_samples = signals.shape[1]

    if mode == "COR": # RAW TPA = S / LP^2 -> No spa substraction
        signals_COR = signals / LP[:, np.newaxis] / LP[:, np.newaxis]
        signals_COR_masked = np.where(mask, signals_COR, np.nan)    
            
        wu_rootfile(root_path, 
                    ["signals_COR", "signals_COR_masked"], 
                    [signals_COR, signals_COR_masked],
                    "Signal")



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
            fig.show()

        

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

    features_names += ["FWHM"]
    features_pars += [FWHM]

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


