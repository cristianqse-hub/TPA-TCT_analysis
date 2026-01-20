import numpy as np

from utils_lib import getVals, wu_rootfile


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
        "TLeft",
        "TRight",
    ]
    features_values = [
        maxV,
        minV,
        BLLevel,
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
    interp_nsamples=20,
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

    TLeft_sig = np.full(n_events, np.nan)
    TRight_sig = np.full(n_events, np.nan)

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