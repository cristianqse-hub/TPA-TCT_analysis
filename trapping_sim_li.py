import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


trapz = getattr(np, "trapezoid", np.trapz)
DEFAULT_N_Z_GRID = 20001

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
    efield = np.full_like(activeVolume_dis, float(fit_parameters["field_offset"]), dtype=float)
    left = activeVolume_dis <= field_center_abs
    efield[left] += float(fit_parameters["field_coeff_mu0"]) * (field_center_abs - activeVolume_dis[left])**2

    epsilon = 1e-10
    mov_e = 1.0 / (
        1.0 + (efield / float(fit_parameters["electron_vsat"]))**float(fit_parameters["electron_beta"])
    )**(1.0 / float(fit_parameters["electron_beta"]))
    mov_h = float(fit_parameters["hole_mu0_factor"]) / (
        1.0
        + (
            float(fit_parameters["hole_mu0_factor"])
            * efield
            / float(fit_parameters["hole_vsat"])
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
    y0 = float(beam_parameters.get("y0", 0.0)) + float(fit_parameters.get("y0", 0.0))
    profile_e = K * profile_e_raw
    profile_h = K * profile_h_raw
    profile_no_offset = profile_e + profile_h
    profile_total = y0 + profile_no_offset

    material_response = {
        "z": activeVolume_dis,
        "efield": efield,
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
