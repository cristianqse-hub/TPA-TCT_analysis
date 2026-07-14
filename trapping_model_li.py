import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.optimize import brentq, least_squares

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


trapz = getattr(np, "trapezoid", np.trapz)
DEFAULT_N_Z_GRID = 20001

DOUBLE_EXP_FIELD_DEFAULT_PARAMETERS = {
    "EF_ExpAmpLeft": {"type": "fixed", "value": 1.0},
    "EF_ExpDecayLeft": {"type": "fixed", "value": 10.0},
    "EF_ExpAmpRight": {"type": "fixed", "value": 1.0},
    "EF_ExpDecayRight": {"type": "fixed", "value": 10.0},
}

MODEL_PARAMETER_NAMES = (
    "BM_z0", "BM_zRight", "BM_zR0", "BM_z_Aberr",
    "BM_CoefA", "BM_CoefB", "BM_area", "BM_scaleAmp", "BM_scaleOffset",
    "MV_beta_e", "MV_vsat_e", "MV_mu0_e",
    "MV_beta_h", "MV_vsat_h", "MV_mu0_h",
    "EF_BiasVoltage", "EF_CoefA", "EF_CoefB", "EF_CoefC", "EF_z0",
    "EF_ExpAmpLeft", "EF_ExpDecayLeft", "EF_ExpAmpRight", "EF_ExpDecayRight",
    "SC_scaleAmp", "TR_tau_e", "TR_tau_h", "SC_scaleOffset", "SC_scale_zShift",
)

PARAMETER_UNITS = {
    "BM_z0": "µm",
    "BM_zRight": "µm",
    "BM_zR0": "µm",
    "BM_z_Aberr": "µm",
    "BM_CoefA": "adim.",
    "BM_CoefB": "1/µm",
    "BM_area": "NE",
    "BM_scaleAmp": "NE",
    "BM_scaleOffset": "NE",
    "MV_beta_e": "adim.",
    "MV_vsat_e": "µm/ns",
    "MV_mu0_e": "µm²/(V ns)",
    "MV_beta_h": "adim.",
    "MV_vsat_h": "µm/ns",
    "MV_mu0_h": "µm²/(V ns)",
    "EF_BiasVoltage": "V",
    "EF_CoefA": "V/µm³",
    "EF_CoefB": "V/µm²",
    "EF_CoefC": "V/µm",
    "EF_z0": "µm",
    "EF_ExpAmpLeft": "V/µm",
    "EF_ExpDecayLeft": "µm",
    "EF_ExpAmpRight": "V/µm",
    "EF_ExpDecayRight": "µm",
    "SC_scaleAmp": "adim.",
    "TR_tau_e": "ns",
    "TR_tau_h": "ns",
    "SC_scaleOffset": "NE",
    "SC_scale_zShift": "µm",
}


def parameter_label(name):
    return f"{name} / {PARAMETER_UNITS.get(name, 'NE')}"

FIT_PARAMETER_NAMES = [
    "electron_beta",
    "hole_beta",
    "hole_mu0_factor",
    "field_center",
    "field_coeff_mu0",
    "field_offset",
    "intensity_scale",
    "tau_e_ns",
    "tau_h_ns",
    "y0",
    "z_shift",
]

FIT_LOWER_BOUNDS = {
    "electron_beta": 0.2,
    "hole_beta": 0.2,
    "hole_mu0_factor": 1e-4,
    "field_center": 0.0,
    "field_coeff_mu0": 0.0,
    "field_offset": 1e-8,
    "intensity_scale": 0.1,
    "tau_e_ns": 1e-6,
    "tau_h_ns": 1e-6,
    "y0": -1.0,
    "z_shift": -50.0,
}

LED_BIAS_CAL = np.array([3.0, 3.10, 3.15, 3.20], dtype=float)
LED_POWER_MW_CAL = np.array([0.234, 1.09, 1.81, 2.60], dtype=float)
LED_POWER_MAX_MW = float(np.max(LED_POWER_MW_CAL))


def zR_aberracion_esferica(z, zR0, z0, a, b=0.0, eps=1e-12):
    z = np.asarray(z, dtype=float)
    dz = np.maximum(z - float(z0), 0.0)
    return np.maximum(float(zR0) + float(a) * dz + float(b) * dz**2, eps)


def lorentz_norm_a_primitiva(z, zc, zR, area=1.0, eps=1e-15):
    z = np.asarray(z, dtype=float)
    x = (z - float(zc)) / float(zR)
    intensidad = 1.0 / (1.0 + x * x)
    amplitud = np.sqrt(np.maximum(intensidad, eps))
    area_amplitud = trapz(amplitud, z)
    if area_amplitud <= 0.0:
        raise ValueError("La integral de la amplitud debe ser positiva")
    amplitud_norm = (float(area) / area_amplitud) * amplitud
    return amplitud_norm**2


def perfil_charge_1mw2(z_eval, z_grid, beam):
    z_eval = np.asarray(z_eval, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)

    z0_step = float(beam["z0_step"])
    z2_end = z0_step + float(beam["z2"])
    active = (z_grid >= z0_step) & (z_grid <= z2_end)
    if not np.any(active):
        raise ValueError("No hay puntos de z_grid dentro del material activo")

    charge = np.empty_like(z_eval, dtype=float)
    for i, zc in enumerate(z_eval):
        zR = zR_aberracion_esferica(
            zc,
            zR0=beam["zR0"],
            z0=beam["z0_ab"],
            a=beam["a"],
            b=beam.get("b", 0.0),
        )
        densidad = lorentz_norm_a_primitiva(z_grid, zc=zc, zR=zR, area=beam["A_area"])
        charge[i] = trapz(densidad[active], z_grid[active])
    return float(beam["y0"]) + float(beam["K"]) * charge, charge


def led_bias_from_name(name):
    match = re.search(r"_LED_([^_]+)", str(name))
    if match is None:
        return np.nan
    value = match.group(1).replace("lamp", "0")
    try:
        return float(value)
    except ValueError:
        return np.nan


def led_power_mw(led_bias):
    if not np.isfinite(led_bias) or led_bias <= 0:
        return 0.0
    return float(np.interp(led_bias, LED_BIAS_CAL, LED_POWER_MW_CAL))


def discover_f2w1_root_files(root_dir="SiC_UVLED/RootFiles"):
    roots = sorted(Path(root_dir).glob("*F2W1*-850V*.root"))
    records = []
    for root in roots:
        led_bias = led_bias_from_name(root.stem)
        power_mw = led_power_mw(led_bias)
        rel_power = power_mw / LED_POWER_MAX_MW if LED_POWER_MAX_MW else 0.0
        records.append(
            {
                "root_file": str(root),
                "label": root.stem,
                "led_bias": float(led_bias),
                "power_mw": float(power_mw),
                "rel_power": float(rel_power),
            }
        )
    records.sort(key=lambda item: (item["rel_power"], item["label"]))
    return records


def f2w1_norm_scale_from_beam_config(beam_config):
    params = beam_config.get("parameters", beam_config)
    metadata = beam_config.get("metadata", {})
    raw_norm = float(params.get("fit_max_norm", 1.0))
    metadata_scale = metadata.get("raw_norm_scale")
    if metadata_scale is not None:
        return float(metadata_scale) * raw_norm
    return raw_norm


def load_f2w1_profile(root_file, norm_scale):
    from utils_lib import getVals

    vals = getVals(root_file, ["Profiles:zSiC", "Profiles:ChargeCSPA_Avr", "Raw:voltage_v"])
    z = np.asarray(vals["Profiles:zSiC"], dtype=float).ravel()
    charge_raw = np.abs(np.asarray(vals["Profiles:ChargeCSPA_Avr"], dtype=float).ravel())
    n_common = min(z.size, charge_raw.size)
    z = z[:n_common]
    charge_raw = charge_raw[:n_common]
    finite = np.isfinite(z) & np.isfinite(charge_raw)
    z = z[finite]
    charge_raw = charge_raw[finite]
    order = np.argsort(z)
    z = z[order]
    charge_raw = charge_raw[order]

    norm = float(norm_scale)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("norm_scale must be finite and positive")
    return z, charge_raw / norm, charge_raw, abs(float(vals["Raw:voltage_v"]))


def ensure_manual_trapping_json(
    path,
    default_parameters,
    profile_index=0,
    steps_per_active_region=400,
    refresh_interval_s=2.0,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    default_parameters = {key: float(value) for key, value in default_parameters.items()}

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("description", "Manual F2W1 trapping-simulation parameters.")
        payload.setdefault("profile_index", int(profile_index))
        payload.setdefault("steps_per_active_region", int(steps_per_active_region))
        payload.setdefault("refresh_interval_s", float(refresh_interval_s))
        params = payload.setdefault("parameters", {})
        changed = False
        for name, value in default_parameters.items():
            if name not in params:
                params[name] = float(value)
                changed = True
        if changed:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        return path

    payload = {
        "description": (
            "Manual F2W1 trapping-simulation parameters. Edit profile_index to choose "
            "the F2W1 profile ordered by increasing relative LED power."
        ),
        "profile_index": int(profile_index),
        "steps_per_active_region": int(steps_per_active_region),
        "refresh_interval_s": float(refresh_interval_s),
        "parameters": default_parameters,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def read_manual_trapping_json(path, default_parameters):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    params = {key: float(value) for key, value in default_parameters.items()}
    params.update(
        {
            key: float(value)
            for key, value in payload.get("parameters", {}).items()
            if key in params
        }
    )
    return {
        "payload": payload,
        "profile_index": int(payload.get("profile_index", 0)),
        "steps_per_active_region": int(payload.get("steps_per_active_region", 400)),
        "refresh_interval_s": max(float(payload.get("refresh_interval_s", 2.0)), 0.2),
        "parameters": params,
    }


def empty_profile_fit_record(profile_index, profile, initial_goodness=1e99):
    return {
        "profile_index": int(profile_index),
        "fit_done": False,
        "goodness_metric": "chi2_dof",
        "chi2_dof": float(initial_goodness),
        "chi2": float(initial_goodness),
        "dof": None,
        "success": False,
        "message": "No fit stored yet.",
        "nfev": 0,
        "residual_calls": 0,
        "max_residual_calls": 0,
        "fixed_sigma": None,
        "fixed_sigma_fraction": None,
        "steps_per_active_region": None,
        "root_file": profile["root_file"],
        "label": profile["label"],
        "led_bias": float(profile["led_bias"]),
        "power_mw": float(profile["power_mw"]),
        "rel_power": float(profile["rel_power"]),
        "initial_parameters": {},
        "fit_parameters": {},
        "fit_errors": {},
        "figure": None,
    }


def ensure_profile_fit_results_json(
    path,
    profiles=None,
    root_dir="SiC_UVLED/RootFiles",
    initial_goodness=1e99,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles = discover_f2w1_root_files(root_dir=root_dir) if profiles is None else list(profiles)

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {
            "description": (
                "Best stored trapping-model fit results for each F2W1 profile index. "
                "Lower chi2_dof is better."
            ),
            "goodness_metric": "chi2_dof",
            "profiles": {},
        }

    payload.setdefault("description", "Best stored trapping-model fit results for each F2W1 profile index.")
    payload.setdefault("goodness_metric", "chi2_dof")
    records = payload.setdefault("profiles", {})

    changed = False
    for index, profile in enumerate(profiles):
        key = str(index)
        if key not in records:
            records[key] = empty_profile_fit_record(index, profile, initial_goodness=initial_goodness)
            changed = True
        else:
            record = records[key]
            for field in ("root_file", "label"):
                if record.get(field) != profile[field]:
                    record[field] = profile[field]
                    changed = True
            for field in ("led_bias", "power_mw", "rel_power"):
                value = float(profile[field])
                if float(record.get(field, np.nan)) != value:
                    record[field] = value
                    changed = True

    if changed or not path.exists():
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    return payload


def make_profile_fit_result_record(
    profile_index,
    profile,
    fit_output,
    fit_result,
    initial_parameters,
    fit_parameters,
    fit_errors,
    fixed_sigma,
    fixed_sigma_fraction,
    steps_per_active_region,
    figure_path=None,
):
    return {
        "profile_index": int(profile_index),
        "fit_done": True,
        "goodness_metric": "chi2_dof",
        "chi2_dof": float(fit_output["chi2_dof"]),
        "chi2": float(fit_output["chi2"]),
        "dof": int(fit_output["dof"]),
        "success": bool(fit_result.success),
        "message": str(fit_result.message),
        "nfev": int(fit_result.nfev),
        "residual_calls": int(fit_output["residual_calls"]),
        "max_residual_calls": int(fit_output["max_residual_calls"]),
        "fixed_sigma": float(fixed_sigma),
        "fixed_sigma_fraction": float(fixed_sigma_fraction),
        "steps_per_active_region": int(steps_per_active_region),
        "root_file": profile["root_file"],
        "label": profile["label"],
        "led_bias": float(profile["led_bias"]),
        "power_mw": float(profile["power_mw"]),
        "rel_power": float(profile["rel_power"]),
        "initial_parameters": {name: float(value) for name, value in initial_parameters.items()},
        "fit_parameters": {name: float(value) for name, value in fit_parameters.items()},
        "fit_errors": {
            name: float(error)
            for name, error in zip(FIT_PARAMETER_NAMES, np.asarray(fit_errors, dtype=float))
        },
        "figure": str(figure_path) if figure_path is not None else None,
    }


def save_profile_fit_result_if_better(
    path,
    profile_index,
    record,
    profiles=None,
    root_dir="SiC_UVLED/RootFiles",
    initial_goodness=1e99,
):
    path = Path(path)
    payload = ensure_profile_fit_results_json(
        path,
        profiles=profiles,
        root_dir=root_dir,
        initial_goodness=initial_goodness,
    )
    key = str(int(profile_index))
    old_record = payload["profiles"].get(key)
    old_goodness = float(old_record.get("chi2_dof", initial_goodness)) if old_record else float(initial_goodness)
    new_goodness = float(record["chi2_dof"])
    updated = new_goodness < old_goodness

    if updated:
        payload["profiles"][key] = record
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    return updated, old_goodness, new_goodness


def append_good_multistart_fit(
    path,
    profile_index,
    profile,
    attempt,
    fit_output,
    fit_result,
    initial_parameters,
    fit_parameters,
    fit_errors,
    fixed_sigma,
    fixed_sigma_fraction,
    steps_per_active_region,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {
            "description": "Accepted multistart trapping-model fits, stored in completion order.",
            "goodness_metric": "chi2_dof",
            "accepted_fits": [],
        }

    payload.setdefault("description", "Accepted multistart trapping-model fits, stored in completion order.")
    payload.setdefault("goodness_metric", "chi2_dof")
    accepted = payload.setdefault("accepted_fits", [])
    stored_iteration = len(accepted)

    record = {
        "stored_iteration": int(stored_iteration),
        "attempt": int(attempt),
        "profile_index": int(profile_index),
        "root_file": profile["root_file"],
        "label": profile["label"],
        "led_bias": float(profile["led_bias"]),
        "power_mw": float(profile["power_mw"]),
        "rel_power": float(profile["rel_power"]),
        "chi2_dof": float(fit_output["chi2_dof"]),
        "chi2": float(fit_output["chi2"]),
        "dof": int(fit_output["dof"]),
        "success": bool(fit_result.success),
        "message": str(fit_result.message),
        "nfev": int(fit_result.nfev),
        "residual_calls": int(fit_output["residual_calls"]),
        "max_residual_calls": int(fit_output["max_residual_calls"]),
        "fixed_sigma": float(fixed_sigma),
        "fixed_sigma_fraction": float(fixed_sigma_fraction),
        "steps_per_active_region": int(steps_per_active_region),
        "initial_parameters": {
            name: float(initial_parameters[name])
            for name in FIT_PARAMETER_NAMES
            if name in initial_parameters
        },
        "fit_parameters": {
            name: float(fit_parameters[name])
            for name in FIT_PARAMETER_NAMES
            if name in fit_parameters
        },
        "fit_errors": {
            name: float(error)
            for name, error in zip(FIT_PARAMETER_NAMES, np.asarray(fit_errors, dtype=float))
        },
        "evaluation_history": fit_output.get("evaluation_history", []),
    }
    accepted.append(record)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return record


def read_good_multistart_fits(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("accepted_fits", [])


def append_multistart_attempt_history(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {
            "description": "All multistart fit attempts and their evaluation histories.",
            "goodness_metric": "chi2_dof",
            "attempts": [],
        }
    payload.setdefault("description", "All multistart fit attempts and their evaluation histories.")
    payload.setdefault("goodness_metric", "chi2_dof")
    attempts = payload.setdefault("attempts", [])
    record = dict(record)
    record["stored_attempt_index"] = len(attempts)
    attempts.append(record)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return record


def evaluate_manual_trapping_control(
    json_path,
    default_parameters,
    beam_config,
    beam_parameters,
    root_dir="SiC_UVLED/RootFiles",
):
    manual_config = read_manual_trapping_json(json_path, default_parameters=default_parameters)
    manual_fit_parameters = dict(default_parameters)
    manual_fit_parameters.update(manual_config["parameters"])
    manual_steps = int(manual_config["steps_per_active_region"])

    profiles = discover_f2w1_root_files(root_dir=root_dir)
    if not profiles:
        raise FileNotFoundError(f"No F2W1 -850V ROOT profiles found in {root_dir}")

    profile_index = int(manual_config["profile_index"])
    if profile_index < 0 or profile_index >= len(profiles):
        raise IndexError(f"profile_index={profile_index} out of range 0..{len(profiles) - 1}")

    profile = profiles[profile_index]
    norm_scale = f2w1_norm_scale_from_beam_config(beam_config)
    z, charge, charge_raw, voltage = load_f2w1_profile(profile["root_file"], norm_scale)

    model, model_e, model_h, model_no_offset, model_offset, response = profile_withTrapps(
        z,
        fit_parameters=manual_fit_parameters,
        beam_parameters=beam_parameters,
        stepsPerAC=manual_steps,
    )
    residual = charge - model
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum((charge - np.mean(charge))**2))
    r2 = 1.0 - float(np.sum(residual**2)) / denom if denom > 0 else float("nan")

    return {
        "config": manual_config,
        "profiles": profiles,
        "profile_index": profile_index,
        "profile": profile,
        "parameters": manual_fit_parameters,
        "steps_per_active_region": manual_steps,
        "z": z,
        "charge": charge,
        "charge_raw": charge_raw,
        "voltage": voltage,
        "model": model,
        "model_e": model_e,
        "model_h": model_h,
        "model_no_offset": model_no_offset,
        "model_offset": model_offset,
        "response": response,
        "residual": residual,
        "rmse": rmse,
        "r2": r2,
    }


def plot_manual_trapping_control(
    result,
    beam_parameters,
    fit_parameter_names=FIT_PARAMETER_NAMES,
    residual_error_fraction=0.0175,
    figsize=(11.0, 6.0),
):
    import matplotlib.pyplot as plt

    params = result["parameters"]
    response = result["response"]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax_profile = fig.add_subplot(gs[:2, :2])
    ax_residual = fig.add_subplot(gs[2, :2], sharex=ax_profile)
    ax_velocity = fig.add_subplot(gs[0, 2])
    ax_response = fig.add_subplot(gs[1, 2])
    ax_field = fig.add_subplot(gs[2, 2])

    z0_plot = beam_parameters["z0"] + params["z_shift"]
    z2_plot = z0_plot + beam_parameters["z2"]
    charge_plot = result["charge"] - result["model_offset"]
    residual_plot = charge_plot - result["model_no_offset"]
    residual_error = float(residual_error_fraction) * float(np.nanmax(np.abs(result["charge"])))
    residual_errors = np.full_like(residual_plot, residual_error, dtype=float)
    residual_dof = max(result["z"].size - len(fit_parameter_names), 1)
    residual_chi2_dof = float(np.sum((residual_plot / residual_errors) ** 2) / residual_dof) if residual_error > 0 else float("nan")
    z_fine = np.linspace(float(np.nanmin(result["z"])), float(np.nanmax(result["z"])), max(5 * result["z"].size, result["z"].size))
    model_fine, model_e_fine, model_h_fine, model_no_offset_fine, _, _ = profile_withTrapps(
        z_fine,
        fit_parameters=params,
        beam_parameters=beam_parameters,
        stepsPerAC=result["steps_per_active_region"],
    )

    ax_profile.plot(result["z"], charge_plot, "o", ms=3, color="0.55", label="F2W1 data")
    ax_profile.plot(
        z_fine,
        model_no_offset_fine,
        color="black",
        lw=2,
        label=f"manual model, R2 = {result['r2']:.3f}",
    )
    ax_profile.plot(z_fine, model_e_fine, color="darkblue", lw=1.5, alpha=0.8, label="electron contribution")
    ax_profile.plot(z_fine, model_h_fine, color="crimson", lw=1.5, alpha=0.8, label="hole contribution")
    ax_profile.axvline(z0_plot, color="tab:green", ls="--", lw=1, label="z0")
    ax_profile.axvline(z2_plot, color="tab:red", ls="--", lw=1, label="z0 + z2")
    ax_profile.set_xlabel("focus position z [um]")
    ax_profile.set_ylabel("charge")
    ax_profile.set_title("Manual JSON fit control")
    ax_profile.legend(frameon=False)
    ax_profile.grid(alpha=0.25)

    ax_residual.errorbar(
        result["z"],
        residual_plot,
        yerr=residual_errors,
        fmt="o",
        ms=3,
        color="0.35",
        ecolor="0.75",
        elinewidth=0.8,
        capsize=0,
        label=f"fixed error = {100 * residual_error_fraction:.2f}% max, chi2/dof = {residual_chi2_dof:.2f}",
    )
    ax_residual.axhline(0.0, color="black", lw=1)
    ax_residual.set_xlabel("focus position z [um]")
    ax_residual.set_ylabel("data - model")
    ax_residual.set_title("residuals")
    ax_residual.legend(frameon=False)
    ax_residual.grid(alpha=0.25)

    ax_velocity.plot(response["z"], response["vdrift_mue"], color="darkblue", label="e")
    ax_velocity.plot(response["z"], response["vdrift_muh"], color="crimson", label="h")
    ax_velocity.set_title("effective drift velocity")
    ax_velocity.set_xlabel("z [um]")
    ax_velocity.set_ylabel("velocity [a.u.]")
    ax_velocity.legend(frameon=False)
    ax_velocity.grid(alpha=0.25)

    ax_response.plot(response["z"], response["response_total"], color="black", label="sum")
    ax_response.plot(response["z"], response["response_e"], color="darkblue", alpha=0.8, label="e")
    ax_response.plot(response["z"], response["response_h"], color="crimson", alpha=0.8, label="h")
    ax_response.set_title("material response")
    ax_response.set_xlabel("z [um]")
    ax_response.legend(frameon=False)
    ax_response.grid(alpha=0.25)

    ax_field.plot(response["z"], response["efield"], color="tab:green")
    ax_field.set_title("effective field")
    ax_field.set_xlabel("z [um]")
    ax_field.set_ylabel("field [a.u.]")
    ax_field.grid(alpha=0.25)

    return fig


def print_manual_trapping_summary(result, json_path, fit_parameter_names=FIT_PARAMETER_NAMES):
    print("Available F2W1 profiles ordered by relative LED power:")
    for idx, profile in enumerate(result["profiles"]):
        marker = "*" if idx == result["profile_index"] else " "
        print(f"{marker} {idx:02d}: rel={profile['rel_power']:.6g}, LED={profile['led_bias']}, {profile['label']}")

    print(f"\nManual JSON: {json_path}")
    print(f"Selected ROOT: {result['profile']['root_file']}")
    print(f"Voltage = {result['voltage']:.6g} V, relative LED power = {result['profile']['rel_power']:.6g}")
    print(f"steps_per_active_region = {result['steps_per_active_region']}")
    print(f"RMSE = {result['rmse']:.6g}")
    print(f"R2 = {result['r2']:.6g}")
    print("\nManual parameters:")
    for name in fit_parameter_names:
        print(f"{name:18s} = {result['parameters'][name]:.8g}")


def watch_manual_trapping_json(
    json_path,
    default_parameters,
    beam_config,
    beam_parameters,
    images_dir="SiC_UVLED/Images",
    study_image_dpi=160,
    root_dir="SiC_UVLED/RootFiles",
    profile_index=0,
    steps_per_active_region=400,
    refresh_interval_s=2.0,
    residual_error_fraction=0.0175,
    max_updates=None,
):
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display

    json_path = Path(json_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    ensure_manual_trapping_json(
        json_path,
        default_parameters=default_parameters,
        profile_index=profile_index,
        steps_per_active_region=steps_per_active_region,
        refresh_interval_s=refresh_interval_s,
    )

    print(f"Watching {json_path}. Interrupt the cell to stop.")
    last_signature = None
    update_count = 0

    while True:
        signature = (json_path.stat().st_mtime_ns, json_path.stat().st_size)
        if signature != last_signature:
            try:
                result = evaluate_manual_trapping_control(
                    json_path,
                    default_parameters=default_parameters,
                    beam_config=beam_config,
                    beam_parameters=beam_parameters,
                    root_dir=root_dir,
                )
                fig = plot_manual_trapping_control(
                    result,
                    beam_parameters=beam_parameters,
                    residual_error_fraction=residual_error_fraction,
                )
                fig_path = images_dir / f"F2W1_manual_trapping_profile_{result['profile_index']:02d}.png"
                fig.savefig(fig_path, dpi=study_image_dpi, bbox_inches="tight")

                clear_output(wait=True)
                print_manual_trapping_summary(result, json_path)
                print(f"\nSaved figure: {fig_path}")
                print(f"Watching for JSON changes every {result['config']['refresh_interval_s']:.2f} s. Interrupt the cell to stop.")
                display(fig)
                plt.close(fig)
                last_signature = signature
                update_count += 1
            except Exception as exc:
                clear_output(wait=True)
                print(f"Manual trapping control failed: {exc}")
                print(f"Fix {json_path} and save it; the cell will retry.")
                last_signature = signature

            if max_updates is not None and update_count >= int(max_updates):
                break

        config = read_manual_trapping_json(json_path, default_parameters=default_parameters)
        time.sleep(float(config["refresh_interval_s"]))


def profile_withTrapps(x_vec, fit_parameters, beam_parameters, stepsPerAC, n_z_grid=DEFAULT_N_Z_GRID):
    x_vec = np.asarray(x_vec, dtype=float)
    z0 = float(beam_parameters["z0"])
    z2 = float(beam_parameters["z2"])
    shift = float(fit_parameters.get("z_shift", 0.0))
    z0_shifted = z0 + shift

    zR_vector = zR_aberracion_esferica(
        x_vec,
        zR0=beam_parameters["zR0"],
        z0=beam_parameters["z0_ab"] + shift,
        a=beam_parameters["a"],
        b=beam_parameters["b"],
    )

    activeVolume_dis = np.linspace(z0_shifted, z0_shifted + z2, int(stepsPerAC))
    dz = activeVolume_dis[1] - activeVolume_dis[0]

    field_center_abs = z0_shifted + float(fit_parameters["field_center"])
    field_coordinate = activeVolume_dis - field_center_abs
    efield = (
        float(fit_parameters["field_a"]) * field_coordinate**2
        + float(fit_parameters["field_b"]) * field_coordinate
        + float(fit_parameters["field_c"])
    )
    if np.any(~np.isfinite(efield)) or np.any(efield <= 0.0):
        raise ValueError("The electric-field polynomial must be finite and positive over the active volume")

    epsilon = 1e-10
    mu0_e = float(fit_parameters["mu0_e"])
    mu0_h = float(fit_parameters["mu0_h"])
    mov_e = mu0_e / (
        1.0
        + (mu0_e * efield / float(fit_parameters["electron_vsat"]))**float(fit_parameters["electron_beta"])
    )**(1.0 / float(fit_parameters["electron_beta"]))
    mov_h = mu0_h / (
        1.0
        + (
            mu0_h * efield / float(fit_parameters["hole_vsat"])
        )**float(fit_parameters["hole_beta"])
    )**(1.0 / float(fit_parameters["hole_beta"]))
    vdrift_mue = mov_e * efield + epsilon
    vdrift_muh = mov_h * efield + epsilon

    mov_time_e = dz / vdrift_mue[:-1]
    mov_time_h = dz / vdrift_muh[:-1]

    diff_survival_e = np.exp(-mov_time_e / float(fit_parameters["tau_e_ns"]))
    diff_survival_h = np.exp(-mov_time_h / float(fit_parameters["tau_h_ns"]))

    response_e = np.zeros_like(activeVolume_dis)
    response_h = np.zeros_like(activeVolume_dis)
    width = activeVolume_dis[-1] - activeVolume_dis[0]
    for i in range(activeVolume_dis.size):
        survival = 1.0
        for j in range(i, diff_survival_e.size):
            response_e[i] += survival * dz / width
            survival *= diff_survival_e[j]

        survival = 1.0
        for j in range(i - 1, -1, -1):
            response_h[i] += survival * dz / width
            survival *= diff_survival_h[j]

    z_grid_gen = np.linspace(x_vec.min() - 100.0, x_vec.max() + 500.0, int(n_z_grid))
    matrix_genProfiles = np.empty((x_vec.size, activeVolume_dis.size), dtype=float)
    for i, (zc, zR) in enumerate(zip(x_vec, zR_vector)):
        gen_profile = lorentz_norm_a_primitiva(
            z_grid_gen, zc=zc, zR=zR, area=beam_parameters["A_area"]
        )
        matrix_genProfiles[i] = np.interp(activeVolume_dis, z_grid_gen, gen_profile)
    matrix_genProfiles *= float(fit_parameters.get("intensity_scale", 1.0))

    profile_e_raw = trapz(matrix_genProfiles * response_e[np.newaxis, :], activeVolume_dis, axis=1)
    profile_h_raw = trapz(matrix_genProfiles * response_h[np.newaxis, :], activeVolume_dis, axis=1)
    K = float(beam_parameters.get("K", 1.0))
    y0 = float(beam_parameters.get("beam_y0", beam_parameters.get("y0", 0.0))) + float(fit_parameters.get("y0", 0.0))
    profile_e = K * profile_e_raw
    profile_h = K * profile_h_raw
    profile_no_offset = profile_e + profile_h
    profile_total = y0 + profile_no_offset

    material_response = {
        "z": activeVolume_dis,
        "efield": efield,
        "field_coordinate": field_coordinate,
        "mobility_e": mov_e,
        "mobility_h": mov_h,
        "vdrift_mue": vdrift_mue,
        "vdrift_muh": vdrift_muh,
        "response_e": response_e,
        "response_h": response_h,
        "response_total": response_e + response_h,
        "zR_vector": zR_vector,
        "matrix_genProfiles": matrix_genProfiles,
        "profile_e_raw": profile_e_raw,
        "profile_h_raw": profile_h_raw,
        "profile_no_offset": profile_no_offset,
        "K": K,
        "y0": y0,
    }

    return profile_total, profile_e, profile_h, profile_no_offset, y0, material_response


def vector_to_fit_parameters(vector, base_parameters, names=FIT_PARAMETER_NAMES):
    params = dict(base_parameters)
    for name, value in zip(names, vector):
        params[name] = float(value)
    return params


def fit_parameters_to_vector(parameters, names=FIT_PARAMETER_NAMES):
    return np.array([float(parameters[name]) for name in names], dtype=float)


def fit_bounds(beam_parameters, names=FIT_PARAMETER_NAMES, lower_overrides=None, upper_overrides=None):
    upper = {
        "electron_beta": 5.0,
        "hole_beta": 5.0,
        "hole_mu0_factor": 2.0,
        "field_center": float(beam_parameters["z2"]),
        "field_coeff_mu0": 50.0,
        "field_offset": 1e5,
        "intensity_scale": 3.0,
        "tau_e_ns": 1e4,
        "tau_h_ns": 1e4,
        "y0": 1.0,
        "z_shift": 50.0,
    }
    lower_dict = dict(FIT_LOWER_BOUNDS)
    if lower_overrides:
        lower_dict.update({name: float(value) for name, value in lower_overrides.items()})
    if upper_overrides:
        upper.update({name: float(value) for name, value in upper_overrides.items()})
    lower = np.array([lower_dict[name] for name in names], dtype=float)
    upper = np.array([upper[name] for name in names], dtype=float)
    return lower, upper


def covariance_from_least_squares(result):
    jac = np.asarray(result.jac, dtype=float)
    n_data, n_params = jac.shape
    dof = max(n_data - n_params, 1)
    chi2_dof = 2.0 * result.cost / dof
    return np.linalg.pinv(jac.T @ jac) * chi2_dof


class SimpleProgressBar:
    def __init__(self, total, desc="progress"):
        self.total = int(total)
        self.desc = desc
        self.n = 0
        self.start = time.time()
        self.last_print = 0
        self.postfix = {}

    def set_postfix(self, values=None, **kwargs):
        if values is None:
            values = {}
        self.postfix = dict(values)
        self.postfix.update(kwargs)

    def _postfix_text(self):
        if not self.postfix:
            return ""
        parts = []
        for key, value in self.postfix.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.4g}")
            else:
                parts.append(f"{key}={value}")
        return ", " + ", ".join(parts)

    def update(self, step=1):
        self.n += int(step)
        now = time.time()
        if self.n == self.total or self.n - self.last_print >= max(1, self.total // 20):
            elapsed = now - self.start
            rate = self.n / elapsed if elapsed > 0 else 0.0
            remaining = max(self.total - self.n, 0)
            eta = remaining / rate if rate > 0 else float("nan")
            print(f"{self.desc}: {self.n}/{self.total} calls, elapsed={elapsed:.1f}s, ETA={eta:.1f}s{self._postfix_text()}")
            self.last_print = self.n

    def close(self):
        elapsed = time.time() - self.start
        if self.n != self.last_print:
            print(f"{self.desc}: {self.n}/{self.total} calls, elapsed={elapsed:.1f}s, done{self._postfix_text()}")


def make_progress_bar(total, desc="residual calls"):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="call")
    return SimpleProgressBar(total=total, desc=desc)


def fit_profile_withTrapps(
    x_vec,
    y_data,
    p0_parameters,
    beam_parameters,
    y_sigma=None,
    stepsPerAC=400,
    max_nfev=150,
    lower_bound_overrides=None,
    upper_bound_overrides=None,
    early_stop_eval_step=-1,
    early_stop_chi2_dof=None,
):
    x_vec = np.asarray(x_vec, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    if y_sigma is None:
        y_sigma = np.ones_like(y_data, dtype=float)
    else:
        y_sigma = np.asarray(y_sigma, dtype=float)

    p0 = fit_parameters_to_vector(p0_parameters)
    lower, upper = fit_bounds(
        beam_parameters,
        lower_overrides=lower_bound_overrides,
        upper_overrides=upper_bound_overrides,
    )
    x_scale = np.maximum(np.abs(p0), 1e-3)

    max_residual_calls = int(max_nfev) * (len(p0) + 1)
    progress = make_progress_bar(max_residual_calls, desc="residual calls")
    eval_counter = {"n": 0}
    dof = max(y_data.size - len(FIT_PARAMETER_NAMES), 1)
    evaluation_history = []
    best_state = {
        "vector": np.array(p0, dtype=float),
        "chi2": float("inf"),
        "chi2_dof": float("inf"),
    }

    class EarlyStopFit(RuntimeError):
        pass

    def residuals(vector):
        if eval_counter["n"] < max_residual_calls:
            progress.update(1)
        eval_counter["n"] += 1
        params = vector_to_fit_parameters(vector, p0_parameters)
        y_model, _, _, _, _, _ = profile_withTrapps(
            x_vec, params, beam_parameters, stepsPerAC=stepsPerAC
        )
        residual = (y_model - y_data) / y_sigma
        chi2 = float(np.sum(residual**2))
        chi2_dof = chi2 / dof
        evaluation_history.append(
            {
                "residual_call": int(eval_counter["n"]),
                "chi2": chi2,
                "chi2_dof": chi2_dof,
            }
        )
        if chi2_dof < best_state["chi2_dof"]:
            best_state["vector"] = np.array(vector, dtype=float)
            best_state["chi2"] = chi2
            best_state["chi2_dof"] = chi2_dof
        if hasattr(progress, "set_postfix"):
            progress.set_postfix({"chi2/dof": chi2_dof, "best": best_state["chi2_dof"]})
        if (
            int(early_stop_eval_step) >= 0
            and eval_counter["n"] == int(early_stop_eval_step)
            and early_stop_chi2_dof is not None
            and chi2_dof > float(early_stop_chi2_dof)
        ):
            raise EarlyStopFit(
                f"Early stop at residual call {eval_counter['n']}: "
                f"chi2/dof={chi2_dof:.6g} > {float(early_stop_chi2_dof):.6g}"
            )
        return residual

    try:
        result = least_squares(
            residuals,
            p0,
            bounds=(lower, upper),
            x_scale=x_scale,
            max_nfev=max_nfev,
            verbose=0,
        )
        early_stopped = False
        early_stop_message = ""
        result_vector = result.x
        covariance = covariance_from_least_squares(result)
    except EarlyStopFit as exc:
        early_stopped = True
        early_stop_message = str(exc)
        result_vector = best_state["vector"]
        result = SimpleNamespace(
            x=result_vector,
            success=False,
            message=early_stop_message,
            nfev=eval_counter["n"],
            cost=0.5 * best_state["chi2"],
            status=-99,
            optimality=np.nan,
        )
        covariance = np.full((len(FIT_PARAMETER_NAMES), len(FIT_PARAMETER_NAMES)), np.nan, dtype=float)
    finally:
        progress.close()

    best_parameters = vector_to_fit_parameters(result_vector, p0_parameters)
    y_fit, y_fit_e, y_fit_h, y_fit_no_offset, y_fit_offset, material_response_fit = profile_withTrapps(
        x_vec, best_parameters, beam_parameters, stepsPerAC=stepsPerAC
    )
    chi2 = float(np.sum(((y_fit - y_data) / y_sigma)**2))

    return {
        "result": result,
        "parameters": best_parameters,
        "covariance": covariance,
        "errors": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2 / dof,
        "evaluation_history": evaluation_history,
        "early_stopped": early_stopped,
        "early_stop_message": early_stop_message,
        "residual_calls": eval_counter["n"],
        "max_residual_calls": max_residual_calls,
        "y_fit": y_fit,
        "y_fit_e": y_fit_e,
        "y_fit_h": y_fit_h,
        "y_fit_no_offset": y_fit_no_offset,
        "y_fit_offset": y_fit_offset,
        "material_response": material_response_fit,
    }


# Configuration-driven fitting API.  This definition intentionally supersedes
# the legacy fixed-parameter-list implementation above.
def load_fit_configuration(configuration):
    """Load and validate a complete model/fit configuration dictionary or JSON file."""
    if isinstance(configuration, (str, Path)):
        path = Path(configuration)
        with path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    elif isinstance(configuration, dict):
        config = json.loads(json.dumps(configuration))
    else:
        raise TypeError("configuration must be a dictionary or a JSON file path")

    if "parameters" not in config and isinstance(config.get("configuration"), dict):
        config = config["configuration"]

    raw_parameters = config.get("parameters")
    if not isinstance(raw_parameters, dict):
        raise ValueError("configuration must contain a 'parameters' dictionary")
    for name, default_spec in DOUBLE_EXP_FIELD_DEFAULT_PARAMETERS.items():
        raw_parameters.setdefault(name, dict(default_spec))
    missing = [name for name in MODEL_PARAMETER_NAMES if name not in raw_parameters]
    unknown = [name for name in raw_parameters if name not in MODEL_PARAMETER_NAMES]
    if missing:
        raise ValueError(f"Missing model parameters: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"Unknown model parameters: {', '.join(unknown)}")

    fit_options = config.setdefault("fit_options", {})
    if not isinstance(fit_options, dict):
        raise TypeError("fit_options must be a dictionary")
    percentual_ranges = bool(fit_options.get("rangeType_percetual", False))
    fit_options["rangeType_percetual"] = percentual_ranges

    normalized = {}
    for name in MODEL_PARAMETER_NAMES:
        specification = raw_parameters[name]
        if not isinstance(specification, dict):
            raise TypeError(f"Parameter '{name}' must be a dictionary")
        has_value = "value" in specification
        has_initial = "initial" in specification
        if not has_value and not has_initial:
            raise ValueError(f"Parameter '{name}' requires 'value'")
        if has_value and has_initial:
            value = float(specification["value"])
            initial_alias = float(specification["initial"])
            if not np.isclose(value, initial_alias, rtol=0.0, atol=0.0):
                raise ValueError(
                    f"Parameter '{name}' has different 'value' and legacy 'initial' values"
                )
        parameter_value = float(
            specification["value"] if has_value else specification["initial"]
        )
        parameter_type = str(specification.get("type", "")).lower()
        if parameter_type == "fixed":
            if not np.isfinite(parameter_value):
                raise ValueError(f"Fixed parameter '{name}' must be finite")
            normalized[name] = {"type": "fixed", "value": parameter_value}
        elif parameter_type == "fit":
            configured_min = (
                float(specification["min"]) if "min" in specification else None
            )
            configured_max = (
                float(specification["max"]) if "max" in specification else None
            )
            configured_bounds = [value for value in (configured_min, configured_max) if value is not None]
            if not np.isfinite(parameter_value) or any(not np.isfinite(value) for value in configured_bounds):
                raise ValueError(f"Fit parameter '{name}' configured values must be finite")
            if percentual_ranges:
                if parameter_value == 0.0 and configured_bounds:
                    raise ValueError(
                        f"Fit parameter '{name}' has value=0 and cannot use percentual ranges"
                    )
                resolved_bounds = [parameter_value * value for value in configured_bounds]
                if configured_min is None:
                    lower = -np.inf
                    upper = resolved_bounds[0] if configured_max is not None else np.inf
                elif configured_max is None:
                    lower = resolved_bounds[0]
                    upper = np.inf
                else:
                    lower, upper = min(resolved_bounds), max(resolved_bounds)
            else:
                lower = -np.inf if configured_min is None else configured_min
                upper = np.inf if configured_max is None else configured_max
            if not lower < upper:
                raise ValueError(f"Fit parameter '{name}' requires min < max")
            if not lower <= parameter_value <= upper:
                raise ValueError(f"Value for '{name}' is outside [min, max]")
            normalized[name] = {"type": "fit", "value": parameter_value}
            if configured_min is not None:
                normalized[name]["min"] = configured_min
            if configured_max is not None:
                normalized[name]["max"] = configured_max
        else:
            raise ValueError(f"Parameter '{name}' type must be 'fit' or 'fixed'")
        if name == "EF_BiasVoltage":
            normalized[name]["enabled"] = bool(specification.get("enabled", False))

    config["parameters"] = normalized
    return config


def _resolved_fit_bounds(parameter_specs, fit_names, fit_options):
    percentual_ranges = bool(fit_options.get("rangeType_percetual", False))
    lower = []
    upper = []
    for name in fit_names:
        specification = parameter_specs[name]
        configured_min = specification.get("min")
        configured_max = specification.get("max")
        if percentual_ranges:
            value = float(specification["value"])
            resolved_min = None if configured_min is None else value * float(configured_min)
            resolved_max = None if configured_max is None else value * float(configured_max)
            if resolved_min is not None and resolved_max is not None:
                lower.append(min(resolved_min, resolved_max))
                upper.append(max(resolved_min, resolved_max))
            else:
                lower.append(-np.inf if resolved_min is None else resolved_min)
                upper.append(np.inf if resolved_max is None else resolved_max)
        else:
            lower.append(-np.inf if configured_min is None else float(configured_min))
            upper.append(np.inf if configured_max is None else float(configured_max))
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _parameter_values(parameter_specs, fit_names=None, fit_vector=None):
    values = {
        name: float(spec["value"])
        for name, spec in parameter_specs.items()
    }
    if fit_names is not None and fit_vector is not None:
        values.update({name: float(value) for name, value in zip(fit_names, fit_vector)})
    values["_EF_BiasVoltage_enabled"] = bool(
        parameter_specs["EF_BiasVoltage"].get("enabled", False)
    )
    return values


def _split_model_parameters(values):
    beam_parameters = {
        "z0": float(values["BM_z0"]),
        "z2": float(values["BM_zRight"]),
        "zR0": float(values["BM_zR0"]),
        "z0_ab": float(values["BM_z_Aberr"]),
        "a": float(values["BM_CoefA"]),
        "b": float(values["BM_CoefB"]),
        "A_area": float(values["BM_area"]),
        "K": float(values["BM_scaleAmp"]),
        "beam_y0": float(values["BM_scaleOffset"]),
        "y0": float(values["BM_scaleOffset"]),
    }
    field_c = float(values["EF_CoefC"])
    if values["_EF_BiasVoltage_enabled"]:
        width = float(values["BM_zRight"])
        center = float(values["EF_z0"])
        u_left = -center
        u_right = width - center
        quadratic_integral = float(values["EF_CoefA"]) * (u_right**3 - u_left**3) / 3.0
        linear_integral = float(values["EF_CoefB"]) * (u_right**2 - u_left**2) / 2.0
        field_c = (
            float(values["EF_BiasVoltage"]) - quadratic_integral - linear_integral
        ) / width

    fit_parameters = {
        "electron_beta": float(values["MV_beta_e"]),
        "electron_vsat": float(values["MV_vsat_e"]),
        "mu0_e": float(values["MV_mu0_e"]),
        "hole_beta": float(values["MV_beta_h"]),
        "hole_vsat": float(values["MV_vsat_h"]),
        "mu0_h": float(values["MV_mu0_h"]),
        "field_a": float(values["EF_CoefA"]),
        "field_b": float(values["EF_CoefB"]),
        "field_c": field_c,
        "field_center": float(values["EF_z0"]),
        "intensity_scale": float(values["SC_scaleAmp"]),
        "tau_e_ns": float(values["TR_tau_e"]),
        "tau_h_ns": float(values["TR_tau_h"]),
        "y0": float(values["SC_scaleOffset"]),
        "z_shift": float(values["SC_scale_zShift"]),
    }
    return beam_parameters, fit_parameters


def simulate_trapping_model(x_vec, configuration):
    """Evaluate the model at fixed values and at initial values of fit parameters."""
    config = load_fit_configuration(configuration)
    values = _parameter_values(config["parameters"])
    beam_parameters, fit_parameters = _split_model_parameters(values)
    options = config["fit_options"]
    return profile_withTrapps(
        np.asarray(x_vec, dtype=float),
        fit_parameters,
        beam_parameters,
        stepsPerAC=int(options.get("steps_per_active_region", 400)),
        n_z_grid=int(options.get("n_z_grid", DEFAULT_N_Z_GRID)),
    )


def load_saved_fit_inputs(json_path):
    """Load all inputs required to reproduce a fit from an exported result JSON."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record.get("configuration"), dict) or not isinstance(record.get("data"), dict):
        raise ValueError("The JSON is not a complete saved-fit input record")
    data = record["data"]
    required = ("x_data", "y_data", "y_sigma")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Saved-fit JSON is missing data arrays: {', '.join(missing)}")
    x_data = np.asarray(data["x_data"], dtype=float)
    y_data = np.asarray(data["y_data"], dtype=float)
    y_sigma = np.asarray(data["y_sigma"], dtype=float)
    if x_data.ndim != 1 or y_data.shape != x_data.shape or y_sigma.shape != x_data.shape:
        raise ValueError("Saved x_data, y_data and y_sigma must be one-dimensional and equal-sized")
    return {
        "record": record,
        "configuration": load_fit_configuration(record["configuration"]),
        "x_data": x_data,
        "y_data": y_data,
        "y_sigma": y_sigma,
        "y_sigma_supplied": bool(data.get("y_sigma_supplied", True)),
    }


def read_fitPars(name_fragments, fits_dir="SiC_UVLED/Fits"):
    """Return saved fit inputs for JSON filenames containing all requested fragments.

    A single match returns ``(x_data, y_data, fit_config, y_sigma)``. Multiple
    matches return a filename-sorted list of those tuples.
    """
    if isinstance(name_fragments, str):
        fragments = [name_fragments]
    else:
        try:
            fragments = [str(fragment) for fragment in name_fragments]
        except TypeError as exc:
            raise TypeError("name_fragments must be a string or an iterable of strings") from exc
    if not fragments or any(not fragment for fragment in fragments):
        raise ValueError("At least one non-empty filename fragment is required")

    directory = Path(fits_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Fit directory does not exist: {directory}")
    matches = sorted(
        path for path in directory.glob("*.json")
        if all(fragment in path.name for fragment in fragments)
    )
    if not matches:
        joined = ", ".join(repr(fragment) for fragment in fragments)
        raise FileNotFoundError(f"No JSON fit filename in {directory} contains all fragments: {joined}")

    results = []
    incomplete = []
    for path in matches:
        try:
            saved = load_saved_fit_inputs(path)
        except ValueError:
            incomplete.append(path.name)
            continue
        results.append((
            saved["x_data"],
            saved["y_data"],
            saved["configuration"],
            saved["y_sigma"],
        ))
    if incomplete:
        raise ValueError(
            "Matched legacy JSON file(s) do not contain x_data, y_data and y_sigma: "
            + ", ".join(incomplete)
        )
    return results[0] if len(results) == 1 else results


def refit_saved_fit(json_path, **fit_options):
    """Re-run a fit directly from a complete JSON exported by a previous fit."""
    saved = load_saved_fit_inputs(json_path)
    record = saved["record"]
    fit_name = fit_options.pop("fit_name", f"{record.get('fit_name', 'fit')}_refit")
    y_sigma = saved["y_sigma"] if saved["y_sigma_supplied"] else None
    return fit_trapping_model(
        saved["x_data"],
        saved["y_data"],
        saved["configuration"],
        y_sigma=y_sigma,
        fit_name=fit_name,
        **fit_options,
    )


def _covariance_from_jacobian(result, scale_by_reduced_chi2):
    jacobian = np.asarray(result.jac, dtype=float)
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    if scale_by_reduced_chi2:
        dof = max(jacobian.shape[0] - jacobian.shape[1], 1)
        covariance *= 2.0 * float(result.cost) / dof
    return 0.5 * (covariance + covariance.T)


def _make_profile_evaluator(
    residual_function, best_vector, lower, upper, fit_names, max_nfev, chi2_min
):
    cache = {}

    def evaluate(index, value):
        value = float(value)
        key = (int(index), float(np.round(value, 14)))
        if key in cache:
            return cache[key]
        nuisance_indices = np.array([i for i in range(best_vector.size) if i != index], dtype=int)
        vector = np.array(best_vector, copy=True)
        vector[index] = value
        if nuisance_indices.size:
            def nuisance_residual(nuisance):
                trial = np.array(vector, copy=True)
                trial[nuisance_indices] = nuisance
                return residual_function(trial)

            nuisance_result = least_squares(
                nuisance_residual,
                best_vector[nuisance_indices],
                bounds=(lower[nuisance_indices], upper[nuisance_indices]),
                x_scale=np.maximum(np.abs(best_vector[nuisance_indices]), 1e-6),
                max_nfev=max_nfev,
            )
            vector[nuisance_indices] = nuisance_result.x
        chi2 = float(np.sum(residual_function(vector) ** 2))
        cache[key] = (chi2, vector)
        return cache[key]

    return evaluate, cache


def _profile_confidence_intervals(
    residual_function, result, covariance, fit_names, lower, upper, max_nfev, points
):
    best = np.asarray(result.x, dtype=float)
    chi2_min = float(np.sum(residual_function(best) ** 2))
    target = chi2_min + 1.0
    evaluate, _ = _make_profile_evaluator(
        residual_function, best, lower, upper, fit_names, max_nfev, chi2_min
    )
    profiles = {}
    errors_minus = np.full(best.size, np.nan)
    errors_plus = np.full(best.size, np.nan)

    for index, name in enumerate(fit_names):
        local_sigma = np.sqrt(max(float(covariance[index, index]), 0.0))
        raw_span = float(upper[index] - lower[index])
        span = raw_span if np.isfinite(raw_span) else max(2.0 * abs(best[index]), 1.0)
        if not np.isfinite(local_sigma) or local_sigma <= 0.0:
            local_sigma = max(span / 100.0, abs(best[index]) * 0.01, 1e-8)

        roots = []
        for direction, boundary in ((-1.0, lower[index]), (1.0, upper[index])):
            previous = float(best[index])
            step = local_sigma
            root = np.nan
            for _ in range(24):
                candidate = float(np.clip(best[index] + direction * step, lower[index], upper[index]))
                delta = evaluate(index, candidate)[0] - target
                if delta >= 0.0:
                    lo, hi = sorted((previous, candidate))
                    try:
                        root = brentq(
                            lambda value: evaluate(index, value)[0] - target,
                            lo, hi, xtol=max(1e-10, span * 1e-8), rtol=1e-8,
                        )
                    except ValueError:
                        root = np.nan
                    break
                if np.isfinite(boundary) and candidate == boundary:
                    break
                previous = candidate
                step *= 1.8
            roots.append(root)

        low_root, high_root = roots
        if np.isfinite(low_root):
            errors_minus[index] = best[index] - low_root
        if np.isfinite(high_root):
            errors_plus[index] = high_root - best[index]

        left_width = errors_minus[index] if np.isfinite(errors_minus[index]) else local_sigma
        right_width = errors_plus[index] if np.isfinite(errors_plus[index]) else local_sigma
        scan_low = max(lower[index], best[index] - 3.0 * left_width)
        scan_high = min(upper[index], best[index] + 3.0 * right_width)
        scan_values = np.unique(np.r_[np.linspace(scan_low, scan_high, max(int(points), 5)), best[index]])
        scan_chi2 = np.array([evaluate(index, value)[0] for value in scan_values])
        profiles[name] = {
            "values": scan_values,
            "chi2": scan_chi2,
            "delta_chi2": scan_chi2 - chi2_min,
            "best": best[index],
            "lower_1sigma": low_root,
            "upper_1sigma": high_root,
        }
    return errors_minus, errors_plus, profiles


def _effective_fit_names(parameter_specs):
    voltage_enabled = bool(parameter_specs["EF_BiasVoltage"].get("enabled", False))
    names = []
    for name in MODEL_PARAMETER_NAMES:
        if parameter_specs[name]["type"] != "fit":
            continue
        if name == "EF_CoefC" and voltage_enabled:
            continue
        if name == "EF_BiasVoltage" and not voltage_enabled:
            continue
        names.append(name)
    return names


def _format_fit_range(lower, upper):
    lower_text = f"{lower:.6g}" if np.isfinite(lower) else "-∞"
    upper_text = f"{upper:.6g}" if np.isfinite(upper) else "+∞"
    return f"[{lower_text}, {upper_text}]"


def _is_near_fit_boundary(value, lower, upper, fraction=1e-3):
    finite_bounds = [bound for bound in (lower, upper) if np.isfinite(bound)]
    finite_span = upper - lower if np.isfinite(lower) and np.isfinite(upper) else np.nan
    for bound in finite_bounds:
        reference = abs(bound)
        if reference == 0.0 and np.isfinite(finite_span):
            reference = abs(finite_span)
        if reference > 0.0 and abs(value - bound) < fraction * reference:
            return True
    return False


def _result_rows(output):
    rows = []
    fitted_index = {name: i for i, name in enumerate(output["fit_names"])}
    voltage_enabled = bool(
        output["configuration"]["parameters"]["EF_BiasVoltage"].get("enabled", False)
    )
    for name in MODEL_PARAMETER_NAMES:
        specification = output["configuration"]["parameters"][name]
        if name == "EF_CoefC" and voltage_enabled:
            parameter_type = "derived"
            initial = output["initial_parameters"][name]
            final = output["parameters"][name]
            error_text = "--"
            relative = np.nan
            range_text = "--"
            near_boundary = False
        elif name == "EF_BiasVoltage" and not voltage_enabled:
            parameter_type = "disabled"
            initial = specification["value"]
            final = initial
            error_text = "--"
            relative = np.nan
            range_text = "--"
            near_boundary = False
        elif specification["type"] == "fit":
            parameter_type = "fit"
            index = fitted_index[name]
            initial = specification["value"]
            final = output["parameters"][name]
            minus = output["errors_minus"][index]
            plus = output["errors_plus"][index]
            covariance_error = np.sqrt(max(float(output["covariance"][index, index]), 0.0))
            lower_bound, upper_bound = output["fit_bounds"][name]
            range_text = _format_fit_range(lower_bound, upper_bound)
            near_boundary = _is_near_fit_boundary(final, lower_bound, upper_bound)
            finite_errors = [value for value in (minus, plus) if np.isfinite(value)]
            if np.isfinite(minus) and np.isfinite(plus):
                error_text = f"-{minus:.4g}/+{plus:.4g}"
                symmetric = float(np.mean([minus, plus]))
            elif np.isfinite(minus):
                error_text = f"-{minus:.4g}/+unresolved (cov: {covariance_error:.4g})"
                symmetric = np.nan
            elif np.isfinite(plus):
                error_text = f"-unresolved (cov: {covariance_error:.4g})/+{plus:.4g}"
                symmetric = np.nan
            else:
                error_text = f"unresolved (cov: {covariance_error:.4g})"
                symmetric = np.nan
            relative = abs(symmetric / final) * 100.0 if final != 0.0 and np.isfinite(symmetric) else np.nan
        else:
            parameter_type = "fixed"
            initial = specification["value"]
            final = specification["value"]
            error_text = "--"
            relative = np.nan
            range_text = "--"
            near_boundary = False
        rows.append([
            parameter_type, name, initial, final, range_text,
            error_text, relative, near_boundary,
        ])
    return rows


def print_fit_summary(output):
    print(f"chi2 / dof = {output['chi2']:.6g} / {output['dof']} = {output['chi2_dof']:.6g}")
    print(f"{'type':8s} {'parameter / unit':34s} {'input value':>13s} {'final':>13s} {'fit range':>22s} {'error (68.3%)':>30s} {'relative':>11s}")
    for parameter_type, name, initial, final, fit_range, error, relative, _ in _result_rows(output):
        relative_text = f"{relative:.3g}%" if np.isfinite(relative) else "--"
        print(f"{parameter_type:8s} {parameter_label(name):34s} {initial:13.6g} {final:13.6g} {fit_range:>22s} {error:>30s} {relative_text:>11s}")


def _table_figure(output):
    import matplotlib.pyplot as plt

    rows = []
    near_boundaries = []
    for parameter_type, name, initial, final, fit_range, error, relative, near_boundary in _result_rows(output):
        rows.append([
            parameter_type, parameter_label(name), f"{initial:.6g}", f"{final:.6g}", fit_range, error,
            f"{relative:.3g}%" if np.isfinite(relative) else "--",
        ])
        near_boundaries.append(near_boundary)
    height = max(8.0, 0.34 * len(rows) + 2.0)
    figure, axis = plt.subplots(figsize=(11.7, height))
    axis.axis("off")
    axis.set_title(
        f"Fit summary — chi2/dof = {output['chi2']:.5g}/{output['dof']} = {output['chi2_dof']:.5g}",
        pad=16,
    )
    table = axis.table(
        cellText=rows,
        colLabels=["type", "parameter / unit", "input value", "final", "fit range", "error (68.3%)", "relative"],
        colWidths=[0.07, 0.24, 0.10, 0.10, 0.17, 0.22, 0.09],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    fit_background = "#fff9d9"
    for row_index, row in enumerate(rows, start=1):
        parameter_type = row[0]
        error_text = row[5]
        if parameter_type == "fit":
            for column_index in range(len(row)):
                table[(row_index, column_index)].set_facecolor(fit_background)
        if "unresolved" in error_text:
            table[(row_index, 5)].get_text().set_color("red")
        if near_boundaries[row_index - 1]:
            table[(row_index, 3)].get_text().set_color("red")
    return figure


def plot_fit_diagnostics(output):
    import matplotlib.pyplot as plt

    x = output["x"]
    response = output["material_response"]
    figure = plt.figure(figsize=(11.0, 6.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 3)
    axis_profile = figure.add_subplot(grid[:2, :2])
    axis_residual = figure.add_subplot(grid[2, :2], sharex=axis_profile)
    axis_velocity = figure.add_subplot(grid[0, 2])
    axis_response = figure.add_subplot(grid[1, 2])
    axis_field = figure.add_subplot(grid[2, 2])

    axis_profile.errorbar(
        x, output["y_data"], yerr=output["y_sigma"], fmt="o", ms=3,
        color="black", ecolor="0.6", elinewidth=0.8, capsize=0, label="simulated data",
    )
    axis_profile.plot(x, output["y_fit"], color="tab:orange", lw=2,
                      label=f"fit, chi2/dof = {output['chi2_dof']:.3f}")
    axis_profile.plot(x, output["y_fit_e"], color="darkblue", lw=1.5, alpha=0.8,
                      label="electron contribution")
    axis_profile.plot(x, output["y_fit_h"], color="crimson", lw=1.5, alpha=0.8,
                      label="hole contribution")
    axis_profile.set(xlabel="focus position z / µm", ylabel="charge / NE",
                     title=f"Fit ID: {output['generation_id']}")
    axis_profile.legend(frameon=False)

    residual = output["y_data"] - output["y_fit"]
    axis_residual.errorbar(
        x, residual, yerr=output["y_sigma"], fmt="o", ms=3,
        color="black", ecolor="0.6", elinewidth=0.8, capsize=0,
        label=f"error, chi2/dof = {output['chi2_dof']:.3f}",
    )
    axis_residual.axhline(0.0, color="black", lw=1)
    axis_residual.set(xlabel="focus position z / µm", ylabel="data - fit / NE", title="residuals")
    axis_residual.legend(frameon=False)

    axis_velocity.plot(response["z"], response["vdrift_mue"], color="darkblue", label="e")
    axis_velocity.plot(response["z"], response["vdrift_muh"], color="crimson", label="h")
    axis_velocity.set(title="effective drift velocity", xlabel="z / µm", ylabel="velocity / (µm/ns)")
    axis_velocity.legend(frameon=False)
    axis_response.plot(response["z"], response["response_total"], color="black", label="sum")
    axis_response.plot(response["z"], response["response_e"], color="darkblue", alpha=0.8, label="e")
    axis_response.plot(response["z"], response["response_h"], color="crimson", alpha=0.8, label="h")
    axis_response.set(title="material response", xlabel="z / µm", ylabel="response / adim.")
    axis_response.legend(frameon=False)
    axis_field.plot(response["z"], response["efield"], color="tab:green")
    axis_field.set(title="electric field", xlabel="z / µm", ylabel="field / (V/µm)")
    for axis in figure.axes:
        axis.grid(alpha=0.25)
    return figure


def plot_fit_correlation(output):
    import matplotlib.pyplot as plt

    correlation = output["correlation"]
    figure, axis = plt.subplots(figsize=(max(7, 0.65 * len(output["fit_names"])), max(6, 0.6 * len(output["fit_names"]))))
    image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    fit_labels = [parameter_label(name) for name in output["fit_names"]]
    axis.set_xticks(range(len(fit_labels)), fit_labels, rotation=60, ha="right")
    axis.set_yticks(range(len(fit_labels)), fit_labels)
    axis.set_title("Fit correlation matrix")
    figure.colorbar(image, ax=axis, label="correlation")
    for row in range(correlation.shape[0]):
        for column in range(correlation.shape[1]):
            value = correlation[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                          color="white" if abs(value) > 0.55 else "black")
    figure.tight_layout()
    return figure


def plot_chi2_profiles(output):
    import matplotlib.pyplot as plt

    names = output["fit_names"]
    columns = 4
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.2 * rows), squeeze=False, constrained_layout=True)
    for axis, name in zip(axes.flat, names):
        profile = output["chi2_profiles"][name]
        axis.plot(profile["values"], profile["delta_chi2"], marker=".", color="black")
        axis.axhline(1.0, color="tab:red", ls="--", label="1 sigma")
        axis.axvline(profile["best"], color="tab:blue", ls="-", label="best")
        if np.isfinite(profile["lower_1sigma"]):
            axis.axvline(profile["lower_1sigma"], color="tab:red", ls=":")
        if np.isfinite(profile["upper_1sigma"]):
            axis.axvline(profile["upper_1sigma"], color="tab:red", ls=":")
        axis.set(
            title=parameter_label(name),
            xlabel=f"parameter / {PARAMETER_UNITS.get(name, 'NE')}",
            ylabel="Delta chi2 / adim.",
        )
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(names):]:
        axis.axis("off")
    return figure


def _json_figure(configuration):
    import matplotlib.pyplot as plt

    text = json.dumps(configuration, indent=2, sort_keys=True)
    lines = text.splitlines()
    pages = []
    for start in range(0, len(lines), 120):
        figure = plt.figure(figsize=(11.7, 8.3))
        figure.text(0.03, 0.97, "Input configuration (reproducible JSON)", va="top", fontsize=12, weight="bold")
        figure.text(0.03, 0.93, "\n".join(lines[start:start + 120]), va="top", family="monospace", fontsize=3.2)
        pages.append(figure)
    return pages


def _save_fit_report(output, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    generation_id = output["generation_id"]
    pdf_path = directory / f"{generation_id}.pdf"
    json_path = directory / f"{generation_id}.json"
    if pdf_path.exists() or json_path.exists():
        raise FileExistsError(f"Fit output already exists for generation ID '{generation_id}'")
    saved_input = {
        "format_version": 1,
        "generation_id": generation_id,
        "fit_name": output["fit_name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "model_library": "trapping_model_li.py",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "configuration": output["configuration"],
        "data": {
            "x_data": np.asarray(output["x"], dtype=float).tolist(),
            "y_data": np.asarray(output["y_data"], dtype=float).tolist(),
            "y_sigma": np.asarray(output["y_sigma"], dtype=float).tolist(),
            "y_sigma_supplied": bool(output["y_sigma_supplied"]),
        },
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(saved_input, stream, indent=2, sort_keys=True)
        stream.write("\n")
    figures = [
        _table_figure(output), plot_fit_diagnostics(output),
        plot_fit_correlation(output), plot_chi2_profiles(output),
        *_json_figure(output["configuration"]),
    ]
    with PdfPages(pdf_path) as pdf:
        for figure in figures:
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return generation_id, pdf_path, json_path


def fit_trapping_model(
    x_vec,
    y_data,
    configuration,
    y_sigma=None,
    *,
    show_summary=False,
    show_fit=False,
    show_covariance=False,
    show_correlation=False,
    show_error_profiles=False,
    save_results=False,
    fit_name="fit",
    output_dir="SiC_UVLED/Fits",
):
    """Fit the trapping model using dynamic fit/fixed parameters and profile-likelihood errors."""
    config = load_fit_configuration(configuration)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(fit_name)).strip("_") or "fit"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_directory = Path(output_dir)
    for _ in range(100):
        generation_id = f"{timestamp}_{random.randint(1, 100)}_{safe_name}"
        if not save_results or not (
            (output_directory / f"{generation_id}.pdf").exists()
            or (output_directory / f"{generation_id}.json").exists()
        ):
            break
    else:
        raise FileExistsError("Could not create a unique fit generation ID")
    specs = config["parameters"]
    options = config["fit_options"]
    x = np.asarray(x_vec, dtype=float)
    data = np.asarray(y_data, dtype=float)
    if x.ndim != 1 or data.shape != x.shape:
        raise ValueError("x_vec and y_data must be one-dimensional arrays with equal shape")
    sigma_supplied = y_sigma is not None
    sigma = np.ones_like(data) if y_sigma is None else np.asarray(y_sigma, dtype=float)
    if sigma.shape != data.shape or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("y_sigma must match y_data and contain finite positive values")

    fit_names = _effective_fit_names(specs)
    if not fit_names:
        raise ValueError("At least one parameter must have type='fit'")
    initial = np.array([specs[name]["value"] for name in fit_names], dtype=float)
    lower, upper = _resolved_fit_bounds(specs, fit_names, options)
    max_nfev = int(options.get("max_nfev", 150))
    profile_max_nfev = int(options.get("profile_max_nfev", max_nfev))
    profile_points = int(options.get("profile_points", 25))
    steps = int(options.get("steps_per_active_region", 400))
    n_z_grid = int(options.get("n_z_grid", DEFAULT_N_Z_GRID))

    def residual(vector):
        try:
            values = _parameter_values(specs, fit_names, vector)
            beam_parameters, fit_parameters = _split_model_parameters(values)
            model = profile_withTrapps(
                x, fit_parameters, beam_parameters, stepsPerAC=steps, n_z_grid=n_z_grid
            )[0]
            result = (model - data) / sigma
            if np.any(~np.isfinite(result)):
                raise ValueError("non-finite residual")
            return result
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            return np.full_like(data, 1e100)

    result = least_squares(
        residual, initial, bounds=(lower, upper),
        x_scale=np.maximum(np.abs(initial), 1e-6), max_nfev=max_nfev,
    )
    absolute_sigma = bool(options.get("absolute_sigma", sigma_supplied))
    covariance = _covariance_from_jacobian(result, scale_by_reduced_chi2=not absolute_sigma)
    errors_minus, errors_plus, profiles = _profile_confidence_intervals(
        residual, result, covariance, fit_names, lower, upper, profile_max_nfev, profile_points
    )
    initial_values = _parameter_values(specs, fit_names, initial)
    initial_beam_parameters, initial_fit_parameters = _split_model_parameters(initial_values)
    initial_values["EF_CoefC"] = initial_fit_parameters["field_c"]
    values = _parameter_values(specs, fit_names, result.x)
    beam_parameters, fit_parameters = _split_model_parameters(values)
    values["EF_CoefC"] = fit_parameters["field_c"]
    y_fit, y_fit_e, y_fit_h, y_fit_no_offset, y_fit_offset, response = profile_withTrapps(
        x, fit_parameters, beam_parameters, stepsPerAC=steps, n_z_grid=n_z_grid
    )
    chi2 = float(np.sum(((y_fit - data) / sigma) ** 2))
    dof = max(data.size - len(fit_names), 1)
    error_stack = np.vstack([errors_minus, errors_plus])
    finite_error_count = np.sum(np.isfinite(error_stack), axis=0)
    symmetric_errors = np.divide(
        np.nansum(error_stack, axis=0),
        finite_error_count,
        out=np.full(errors_minus.shape, np.nan),
        where=finite_error_count > 0,
    )
    covariance_diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    covariance_scale = np.outer(covariance_diagonal, covariance_diagonal)
    correlation = np.divide(
        covariance,
        covariance_scale,
        out=np.full_like(covariance, np.nan),
        where=covariance_scale > 0.0,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    output = {
        "result": result, "generation_id": generation_id, "fit_name": safe_name,
        "configuration": config, "initial_parameters": initial_values, "parameters": values,
        "fit_names": fit_names,
        "fit_bounds": {
            name: (float(lower[index]), float(upper[index]))
            for index, name in enumerate(fit_names)
        },
        "covariance": covariance, "correlation": correlation,
        "errors": symmetric_errors, "errors_minus": errors_minus, "errors_plus": errors_plus,
        "chi2_profiles": profiles, "chi2": chi2, "dof": dof, "chi2_dof": chi2 / dof,
        "x": x, "y_data": data, "y_sigma": sigma,
        "y_sigma_supplied": sigma_supplied,
        "y_fit": y_fit, "y_fit_e": y_fit_e, "y_fit_h": y_fit_h,
        "y_fit_no_offset": y_fit_no_offset, "y_fit_offset": y_fit_offset,
        "material_response": response,
    }
    if show_summary:
        print_fit_summary(output)
    if show_fit or show_covariance or show_correlation or show_error_profiles:
        import matplotlib.pyplot as plt
        if show_fit:
            plot_fit_diagnostics(output)
            plt.show()
        if show_covariance or show_correlation:
            plot_fit_correlation(output)
            plt.show()
        if show_error_profiles:
            plot_chi2_profiles(output)
            plt.show()
    if save_results:
        generation_id, pdf_path, json_path = _save_fit_report(output, output_dir)
        output.update({"pdf_path": pdf_path, "json_path": json_path})
    return output


def fit_profile_withTrapps(x_vec, y_data, configuration, y_sigma=None, **kwargs):
    """Compatibility name for the configuration-driven fitting API."""
    return fit_trapping_model(x_vec, y_data, configuration, y_sigma=y_sigma, **kwargs)
