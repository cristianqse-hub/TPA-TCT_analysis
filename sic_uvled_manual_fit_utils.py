from pathlib import Path
import json
import re
import time

import numpy as np
import matplotlib.pyplot as plt
import ROOT
from ROOT import gROOT
from IPython.display import clear_output, display

from utils_lib import getVals

gROOT.SetBatch(ROOT.kTRUE)

trapz = getattr(np, "trapezoid", np.trapz)

BASE_DIR = Path("SiC_UVLED")
ROOT_DIR = BASE_DIR / "RootFiles"
CONFIG_DIR = BASE_DIR / "configs"

LED_BIAS_CAL = np.array([3.0, 3.10, 3.15, 3.20], dtype=float)
LED_POWER_MW_CAL = np.array([0.234, 1.09, 1.81, 2.60], dtype=float)
LED_POWER_MAX_MW = float(np.max(LED_POWER_MW_CAL))

nominal_mobility_params = {
    "electron_mu0": 950.0,
    "electron_vsat": 2.0e7,
    "electron_beta": 1.15,
    "hole_mu0": 120.0,
    "hole_vsat": 1.2e7,
    "hole_beta": 1.25,
}
mobility_names = list(nominal_mobility_params)


def led_bias_from_name(name):
    match = re.search(r"_LED_([^_]+)", name)
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


def discover_f2w1_root_files(root_dir=ROOT_DIR):
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


def discover_1mw2_root_files(root_dir=ROOT_DIR):
    roots = sorted(Path(root_dir).glob("*1MW2*.root"))
    return [{"root_file": str(root), "label": root.stem} for root in roots]


def default_1mw2_beam_params():
    return {
        "z0_step": -2.561761988852449,
        "zR0": 4.385374879802671,
        "a": 0.004461681253026355,
        "b": 0.0017325631823942112,
        "z2": 50.337266961172645,
        "z0_ab_frac": 0.3016999164441918,
        "A_area": 9.106901819568089,
        "K": 1.9048051801350496,
        "y0": -0.031227930105467934,
    }


def read_1mw2_beam_json(path=CONFIG_DIR / "1MW2_beam_fit_params.json"):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    params = raw.get("parameters", raw)
    metadata = raw.get("metadata", {})
    out = default_1mw2_beam_params()
    out.update({key: float(params[key]) for key in out if key in params})
    out["z0_ab"] = float(params.get("z0_ab", out["z0_step"] + out["z0_ab_frac"] * out["z2"]))
    out["z2_end"] = float(params.get("z2_end", out["z0_step"] + out["z2"]))
    out["fit_max_norm"] = float(params.get("fit_max_norm", 1.0))
    if "raw_norm_scale" in metadata:
        out["raw_norm_scale"] = float(metadata["raw_norm_scale"])
    return out


def load_1mw2_norm_scale(beam_params):
    if "fit_max_norm" not in beam_params:
        return 1.0
    raw_norm = float(beam_params.get("fit_max_norm", 1.0))
    metadata_scale = beam_params.get("raw_norm_scale")
    if metadata_scale is not None:
        return float(metadata_scale) * raw_norm
    return raw_norm


def load_profile(root_file, norm_scale):
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


def load_1mw2_beam_profile(root_file):
    vals = getVals(root_file, ["Profiles:zSiC", "Profiles:ChargeCSPA_Avr"])
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
    norm = float(np.nanmax(charge_raw))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("1MW2 raw charge maximum must be finite and positive")
    return z, charge_raw / norm, charge_raw, norm


def profile_peak_fwhm(z, y):
    z = np.asarray(z, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    finite = np.isfinite(z) & np.isfinite(y)
    z = z[finite]
    y = y[finite]
    if z.size < 3:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(z)
    z = z[order]
    y = y[order]
    peak = float(np.nanmax(y))
    baseline = float(np.nanmin(y))
    half = baseline + 0.5 * (peak - baseline)
    crossings = []
    diff = y - half
    for i in range(z.size - 1):
        if diff[i] == 0.0:
            crossings.append(float(z[i]))
        if diff[i] * diff[i + 1] < 0.0:
            frac = -diff[i] / (diff[i + 1] - diff[i])
            crossings.append(float(z[i] + frac * (z[i + 1] - z[i])))
    if diff[-1] == 0.0:
        crossings.append(float(z[-1]))
    if len(crossings) < 2:
        return peak, half, float("nan")
    return peak, half, float(crossings[-1] - crossings[0])


def find_left_rising_z50(z, y):
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    level = float(np.nanmin(y)) + 0.5 * float(np.nanmax(y) - np.nanmin(y))
    peak_idx = int(np.nanargmax(y))
    left = y[:peak_idx + 1]
    left_z = z[:peak_idx + 1]
    crossings = np.where((left[:-1] <= level) & (left[1:] >= level))[0]
    if crossings.size == 0:
        return float(left_z[int(np.nanargmin(np.abs(left - level)))])
    i = int(crossings[-1])
    y0, y1 = left[i], left[i + 1]
    z0, z1 = left_z[i], left_z[i + 1]
    if y1 == y0:
        return float(z0)
    return float(z0 + (level - y0) * (z1 - z0) / (y1 - y0))


class ManualFitContext:
    def __init__(self, beam_params, root_records, profile_index, norm_scale):
        self.beam_params = dict(beam_params)
        self.root_records = list(root_records)
        self.profile_index = int(profile_index)
        self.profile_record = self.root_records[self.profile_index]
        self.z, self.charge, self.charge_raw, self.voltage = load_profile(self.profile_record["root_file"], norm_scale)
        self.z0_initial = find_left_rising_z50(self.z, self.charge)

        self.beam_z0_step_ref = float(self.beam_params["z0_step"])
        self.beam_zR0 = float(self.beam_params["zR0"])
        self.beam_a = float(self.beam_params["a"])
        self.beam_b = float(self.beam_params["b"])
        self.beam_z2_width = float(self.beam_params["z2"])
        self.beam_z0_ab_ref = float(self.beam_params["z0_ab"])
        self.beam_A_area = float(self.beam_params["A_area"])
        self.fixed_K = float(self.beam_params["K"]) / float(self.beam_params.get("fit_max_norm", 1.0))

        dx = 0.5
        self.z_material = np.arange(0.0, self.beam_z2_width, dx, dtype=float)
        if self.z_material.size == 0 or self.z_material[-1] < self.beam_z2_width:
            self.z_material = np.append(self.z_material, self.beam_z2_width)
        self.material_segment_um = np.diff(self.z_material)
        self.material_points = self.z_material.size
        self.material_width_um = float(self.beam_z2_width)

        z_shift_initial = self.z0_initial - self.beam_z0_step_ref
        z_shift_bounds = (z_shift_initial - 80.0, z_shift_initial + 80.0)
        self.z_grid = np.linspace(
            np.nanmin(self.z) + z_shift_bounds[0] - 150.0,
            np.nanmax(self.z) + z_shift_bounds[1] + 550.0,
            8001,
        )

    def shifted_beam_geometry(self, z_shift, beam_width_scale=1.0):
        z0_step = self.beam_z0_step_ref + float(z_shift)
        width = self.beam_z2_width * float(beam_width_scale)
        return {
            "z0_step": z0_step,
            "z0_ab": self.beam_z0_ab_ref + float(z_shift),
            "z2_end": z0_step + width,
            "width": width,
        }

    def zR_aberracion_esferica(self, z, zR0, z0, a, b, eps=1e-12):
        z = np.asarray(z, dtype=float)
        dz = np.maximum(z - z0, 0.0)
        return np.maximum(zR0 + a * dz + b * dz**2, eps)

    def lorentz_norm_a_primitiva(self, z, zR, zc=0.0, A_area=1.0, eps=1e-15):
        z = np.asarray(z, dtype=float)
        zR = float(zR)
        if zR <= 0:
            raise ValueError("zR must be > 0")
        x = (z - zc) / zR
        L_raw = 1.0 / (1.0 + x * x)
        A_raw = np.sqrt(np.maximum(L_raw, eps))
        area_A = trapz(A_raw, z)
        if area_A <= 0:
            raise ValueError("A_raw area is not positive")
        A_norm = (A_area / area_A) * A_raw
        return A_norm**2, A_norm, L_raw

    def mobility_values_from_fit(self, values):
        return {name: float(values.get(name, nominal_mobility_params[name])) for name in mobility_names}

    def get_driftVel(self, e_field, carrier, mobility_params=None):
        E = np.asarray(e_field, dtype=float) * 1e4
        params = dict(nominal_mobility_params)
        if mobility_params is not None:
            params.update({key: float(value) for key, value in mobility_params.items() if key in params})
        if carrier.lower() == "electron":
            mu0, vsat, beta = params["electron_mu0"], params["electron_vsat"], params["electron_beta"]
        elif carrier.lower() == "hole":
            mu0, vsat, beta = params["hole_mu0"], params["hole_vsat"], params["hole_beta"]
        else:
            raise ValueError("carrier must be 'electron' or 'hole'")
        return (mu0 * E) / (1 + (mu0 * E / vsat)**beta)**(1 / beta)

    def drift_velocity_um_per_ns(self, e_field, carrier, mobility_params=None):
        return self.get_driftVel(e_field, carrier, mobility_params=mobility_params) * 1e-5

    def induced_path_response(self, P_decay, v_drift, injection_index, direction):
        survival = 1.0
        response = 0.0
        segment_indices = range(injection_index, self.material_points - 1) if direction > 0 else range(injection_index, 0, -1)
        for k in segment_indices:
            segment_index = k if direction > 0 else k - 1
            if v_drift[segment_index] <= 0.0:
                break
            segment_um = self.material_segment_um[segment_index]
            response += survival * segment_um / self.material_width_um
            survival *= 1.0 - P_decay[segment_index]
        return response

    def compute_trapping_response(self, e_field, tau_e_ns, tau_h_ns, mobility_params=None):
        mobility_params = self.mobility_values_from_fit(mobility_params or {})
        v_drift_e = self.drift_velocity_um_per_ns(e_field, "electron", mobility_params=mobility_params)
        v_drift_h = self.drift_velocity_um_per_ns(e_field, "hole", mobility_params=mobility_params)
        P_decay_e = 1 - np.exp(-0.5 / np.clip(v_drift_e * float(tau_e_ns), 1e-12, None))
        P_decay_h = 1 - np.exp(-0.5 / np.clip(v_drift_h * float(tau_h_ns), 1e-12, None))
        resp_e = np.zeros(self.material_points)
        resp_h = np.zeros(self.material_points)
        for j in range(self.material_points):
            resp_e[j] = self.induced_path_response(P_decay_e, v_drift_e, j, direction=1)
            resp_h[j] = self.induced_path_response(P_decay_h, v_drift_h, j, direction=-1)
        return resp_e + resp_h, resp_e, resp_h, {"electron_um_per_ns": v_drift_e, "hole_um_per_ns": v_drift_h}

    def compute_parabolic_field(self, field_center, field_coeff, field_offset):
        center = max(float(field_center), 0.0)
        coeff = max(float(field_coeff), 0.0)
        offset = max(float(field_offset), 1e-12)
        dz_left = np.maximum(center - self.z_material, 0.0)
        raw = offset + coeff * dz_left**2
        integral = trapz(raw, self.z_material)
        if integral <= 0.0:
            raise ValueError("raw electric-field integral must be positive")
        return raw * (self.voltage / integral)

    def compute_lineal_field(self, field_z0_value, field_slope):
        z0_value = max(float(field_z0_value), 1e-12)
        slope = min(float(field_slope), 0.0)
        raw = np.clip(z0_value + slope * self.z_material, 0.0, None)
        integral = trapz(raw, self.z_material)
        if integral <= 0.0:
            raise ValueError("raw electric-field integral must be positive")
        return raw * (self.voltage / integral)

    def model_components(self, values, field_kind):
        beam_width_scale = float(values.get("beam_width_scale", 1.0))
        geom = self.shifted_beam_geometry(values["z_shift"], beam_width_scale=beam_width_scale)
        active = (self.z_grid >= geom["z0_step"]) & (self.z_grid <= geom["z2_end"])
        mobility = self.mobility_values_from_fit(values)
        if field_kind == "parabolic":
            e_field = self.compute_parabolic_field(values["field_center"], values["field_coeff"], values["field_offset"])
        elif field_kind == "lineal":
            e_field = self.compute_lineal_field(values["field_z0_value"], values["field_slope"])
        else:
            raise ValueError("field_kind must be 'parabolic' or 'lineal'")
        response, response_e, response_h, drift_vel = self.compute_trapping_response(
            e_field, values["tau_e_ns"], values["tau_h_ns"], mobility_params=mobility
        )
        z_response = geom["z0_step"] + self.z_material
        total_on_u = np.interp(self.z_grid, z_response, response, left=0.0, right=0.0)
        e_on_u = np.interp(self.z_grid, z_response, response_e, left=0.0, right=0.0)
        h_on_u = np.interp(self.z_grid, z_response, response_h, left=0.0, right=0.0)
        q_total = np.empty_like(self.z)
        q_e = np.empty_like(self.z)
        q_h = np.empty_like(self.z)
        for i, zc in enumerate(self.z):
            zR_eff = self.zR_aberracion_esferica(
                zc, zR0=self.beam_zR0, z0=geom["z0_ab"], a=self.beam_a, b=self.beam_b
            )
            L, _, _ = self.lorentz_norm_a_primitiva(self.z_grid, zR=zR_eff, zc=zc, A_area=self.beam_A_area)
            q_total[i] = trapz((L * total_on_u)[active], self.z_grid[active])
            q_e[i] = trapz((L * e_on_u)[active], self.z_grid[active])
            q_h[i] = trapz((L * h_on_u)[active], self.z_grid[active])
        K = self.fixed_K * float(values.get("intensity_scale", 1.0))
        y0 = float(values.get("y0", 0.0))
        return y0 + K * q_total, K * q_e, K * q_h, response, response_e, response_h, e_field, drift_vel, geom


PARABOLIC_NAMES = [
    "tau_e_ns", "tau_h_ns", "field_center", "field_coeff", "field_offset",
    "intensity_scale", "beam_width_scale", *mobility_names, "y0", "z_shift",
]
LINEAL_NAMES = [
    "tau_e_ns", "tau_h_ns", "field_z0_value", "field_slope",
    "intensity_scale", "beam_width_scale", *mobility_names, "y0", "z_shift",
]


def default_parameters(field_kind):
    common = {
        "tau_e_ns": 0.05,
        "tau_h_ns": 0.002,
        "intensity_scale": 1.0,
        "beam_width_scale": 1.0,
        **nominal_mobility_params,
        "y0": 0.0,
        "z_shift": 0.0,
    }
    if field_kind == "parabolic":
        common.update({"field_center": 30.0, "field_coeff": 0.01, "field_offset": 0.5})
    elif field_kind == "lineal":
        common.update({"field_z0_value": 1.0, "field_slope": -0.05})
    else:
        raise ValueError("field_kind must be 'parabolic' or 'lineal'")
    return common


def ensure_manual_json(path, field_kind, profile_index=0, refresh_interval_s=2.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = PARABOLIC_NAMES if field_kind == "parabolic" else LINEAL_NAMES
    defaults = default_parameters(field_kind)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("profile_index", int(profile_index))
        payload.setdefault("refresh_interval_s", float(refresh_interval_s))
        params = payload.setdefault("parameters", {})
        changed = False
        for name in names:
            if name not in params:
                params[name] = float(defaults[name])
                changed = True
        if changed or "profile_index" not in payload:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        return path
    payload = {
        "description": f"Manual F2W1 {field_kind} field model parameters.",
        "profile_index": int(profile_index),
        "refresh_interval_s": float(refresh_interval_s),
        "parameters": {name: float(defaults[name]) for name in names},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def read_manual_json(path, field_kind):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    names = PARABOLIC_NAMES if field_kind == "parabolic" else LINEAL_NAMES
    params = default_parameters(field_kind)
    params.update({key: float(value) for key, value in payload.get("parameters", {}).items() if key in params})
    return {
        "payload": payload,
        "profile_index": int(payload.get("profile_index", payload.get("metadata", {}).get("profile_index", 0))),
        "refresh_interval_s": max(float(payload.get("refresh_interval_s", 2.0)), 0.2),
        "parameters": {name: float(params[name]) for name in names},
        "parameter_names": names,
    }


def evaluate_manual_model(context, values, field_kind):
    charge, charge_e, charge_h, response, response_e, response_h, e_field, drift_vel, geom = context.model_components(values, field_kind)
    residual = context.charge - charge
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum((context.charge - np.mean(context.charge))**2))
    r2 = 1.0 - float(np.sum(residual**2)) / denom if denom > 0 else float("nan")
    data_peak, _, data_fwhm = profile_peak_fwhm(context.z, context.charge)
    model_peak, _, model_fwhm = profile_peak_fwhm(context.z, charge)
    return {
        "charge": charge,
        "charge_e": charge_e,
        "charge_h": charge_h,
        "response": response,
        "response_e": response_e,
        "response_h": response_h,
        "e_field": e_field,
        "drift_vel": drift_vel,
        "geom": geom,
        "residual": residual,
        "rmse": rmse,
        "r2": r2,
        "data_peak": data_peak,
        "data_fwhm": data_fwhm,
        "model_peak": model_peak,
        "model_fwhm": model_fwhm,
    }


def plot_manual_control(context, values, field_kind, json_path):
    result = evaluate_manual_model(context, values, field_kind)
    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.0, 0.95], height_ratios=[1.0, 1.0, 0.65])
    ax_profile = fig.add_subplot(gs[0:2, 0:2])
    ax_resid = fig.add_subplot(gs[2, 0:2], sharex=ax_profile)
    ax_response = fig.add_subplot(gs[0, 2])
    ax_field = fig.add_subplot(gs[1, 2])
    ax_velocity = fig.add_subplot(gs[2, 2])

    ax_profile.plot(context.z, context.charge, ".", color="black", label="data / max(1MW2)")
    ax_profile.plot(context.z, result["charge"], "-", color="tab:red", label=f"{field_kind} JSON model")
    ax_profile.plot(context.z, result["charge_e"], "-", color="darkblue", alpha=0.75, label="electrons")
    ax_profile.plot(context.z, result["charge_h"], "-", color="crimson", alpha=0.75, label="holes")
    ax_profile.axvline(context.z0_initial, linestyle=":", color="tab:green", label="z0 initial 50%")
    ax_profile.axvline(result["geom"]["z0_step"], linestyle="--", color="gray", label="z0 model")
    ax_profile.axvline(result["geom"]["z2_end"], linestyle="--", color="black", label="z0 + width")
    ax_profile.set_ylabel("charge / max(1MW2)")
    ax_profile.grid(alpha=0.25)
    ax_profile.legend(loc="best", fontsize=8)
    ax_profile.set_title(
        f"F2W1 idx={context.profile_index}, rel={context.profile_record['rel_power']:.4g}, "
        f"{field_kind}: RMSE={result['rmse']:.5g}, R2={result['r2']:.6g}; "
        f"peak {result['model_peak']:.4g}/{result['data_peak']:.4g}, "
        f"FWHM {result['model_fwhm']:.4g}/{result['data_fwhm']:.4g}"
    )

    ax_resid.plot(context.z, result["residual"], ".-", color="tab:purple", label="data - model")
    ax_resid.axhline(0.0, color="black", lw=1)
    ax_resid.set_xlabel("z (SiC) [um]")
    ax_resid.set_ylabel("residual")
    ax_resid.grid(alpha=0.25)
    ax_resid.legend(loc="best", fontsize=8)

    ax_response.plot(context.z_material, result["response"], color="tab:blue", label="total")
    ax_response.plot(context.z_material, result["response_e"], color="darkblue", alpha=0.8, label="e")
    ax_response.plot(context.z_material, result["response_h"], color="crimson", alpha=0.8, label="h")
    ax_response.set_title("response")
    ax_response.set_xlabel("z material [um]")
    ax_response.grid(alpha=0.25)
    ax_response.legend(loc="best", fontsize=8)

    ax_field.plot(context.z_material, result["e_field"], color="tab:green")
    if field_kind == "parabolic":
        ax_field.axvline(values["field_center"], linestyle="--", color="tab:purple", label="field_center")
        ax_field.legend(loc="best", fontsize=8)
    ax_field.set_title("E field")
    ax_field.set_xlabel("z material [um]")
    ax_field.set_ylabel("E [V/um]")
    ax_field.grid(alpha=0.25)

    ax_velocity.plot(context.z_material, result["drift_vel"]["electron_um_per_ns"], color="darkblue", label="v_e")
    ax_velocity.plot(context.z_material, result["drift_vel"]["hole_um_per_ns"], color="crimson", label="v_h")
    ax_velocity.set_title("drift velocity")
    ax_velocity.set_xlabel("z material [um]")
    ax_velocity.set_ylabel("um/ns")
    ax_velocity.grid(alpha=0.25)
    ax_velocity.legend(loc="best", fontsize=8)

    fig.suptitle(str(json_path), fontsize=10, y=0.995)
    fig.tight_layout()
    return fig, result


def watch_manual_json(path, field_kind, beam_json_path=CONFIG_DIR / "1MW2_beam_fit_params.json"):
    ensure_manual_json(path, field_kind)
    root_records = discover_f2w1_root_files()
    if not root_records:
        raise RuntimeError(f"No F2W1 ROOT files found in {ROOT_DIR}")
    beam_params = read_1mw2_beam_json(beam_json_path)
    norm_scale = float(beam_params.get("raw_norm_scale", beam_params.get("fit_max_norm", 1.0)))
    if norm_scale <= 0:
        norm_scale = 1.0

    last_signature = None
    refresh_s = 2.0
    while True:
        try:
            path = Path(path)
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != last_signature:
                config = read_manual_json(path, field_kind)
                refresh_s = config["refresh_interval_s"]
                profile_index = config["profile_index"]
                if profile_index < 0 or profile_index >= len(root_records):
                    raise IndexError(f"profile_index={profile_index} out of range 0..{len(root_records) - 1}")
                context = ManualFitContext(beam_params, root_records, profile_index, norm_scale=norm_scale)

                clear_output(wait=True)
                print("Manual parameter JSON:", path)
                print("Field model:", field_kind)
                print("Refresh interval [s]:", refresh_s)
                print("Beam JSON:", beam_json_path)
                print("Profile index:", context.profile_index)
                print("Profile root:", context.profile_record["root_file"])
                print("Relative LED power:", context.profile_record["rel_power"])
                print("Fixed from 1MW2 beam: zR0, z0_step, z0_ab, z2, a, b, A_area")
                print("K_eff = K_1MW2 / fit_max_norm * intensity_scale")
                for name in config["parameter_names"]:
                    print(f"{name:>15s} = {config['parameters'][name]:.8g}")
                fig, result = plot_manual_control(context, config["parameters"], field_kind, path)
                print(f"RMSE={result['rmse']:.8g}, R2={result['r2']:.8g}")
                print(f"Data/Fit peak: {result['data_peak']:.8g} / {result['model_peak']:.8g}")
                print(f"Data/Fit FWHM: {result['data_fwhm']:.8g} / {result['model_fwhm']:.8g}")
                display(fig)
                plt.close(fig)
                last_signature = signature
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            clear_output(wait=True)
            print("Error while reading/plotting manual JSON:", exc)
            print("JSON path:", path)
        time.sleep(refresh_s)


ONE_MW2_LINEAL_NAMES = ["z0_step", "zR0", "a", "z2", "z0_ab_frac", "A_area", "K", "y0"]


def zR_aberracion_esferica(z, zR0, z0, a, b=0.0, eps=1e-12):
    z = np.asarray(z, dtype=float)
    dz = np.maximum(z - float(z0), 0.0)
    return np.maximum(float(zR0) + float(a) * dz + float(b) * dz**2, eps)


def lorentz_norm_a_primitiva(z, zR, zc=0.0, A_area=1.0, eps=1e-15):
    z = np.asarray(z, dtype=float)
    zR = float(zR)
    if zR <= 0:
        raise ValueError("zR must be > 0")
    x = (z - float(zc)) / zR
    L_raw = 1.0 / (1.0 + x * x)
    A_raw = np.sqrt(np.maximum(L_raw, eps))
    area_A = trapz(A_raw, z)
    if area_A <= 0:
        raise ValueError("A_raw area is not positive")
    A_norm = (float(A_area) / area_A) * A_raw
    return A_norm**2


def perfil_carga_sep_z0(z_eval, z_grid, zR0, z0_step, z2, z0_ab, a, b, A_area):
    z_eval = np.asarray(z_eval, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    active = (z_grid >= float(z0_step)) & (z_grid <= float(z0_step) + float(z2))
    if not np.any(active):
        raise ValueError("No active points inside z0_step..z0_step+z2")
    charge = np.empty_like(z_eval)
    for i, zc in enumerate(z_eval):
        zR_eff = zR_aberracion_esferica(zc, zR0=zR0, z0=z0_ab, a=a, b=b)
        L = lorentz_norm_a_primitiva(z_grid, zR=zR_eff, zc=zc, A_area=A_area)
        charge[i] = trapz(L[active], z_grid[active])
    return charge


def default_1mw2_lineal_parameters(beam_json_path=CONFIG_DIR / "1MW2_beam_fit_params.json"):
    beam = read_1mw2_beam_json(beam_json_path)
    return {name: float(beam[name]) for name in ONE_MW2_LINEAL_NAMES}


def ensure_1mw2_lineal_json(
    path,
    beam_json_path=CONFIG_DIR / "1MW2_beam_fit_params.json",
    root_index=0,
    refresh_interval_s=2.0,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = default_1mw2_lineal_parameters(beam_json_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        changed = False
        if "root_index" not in payload:
            payload["root_index"] = int(root_index)
            changed = True
        if "refresh_interval_s" not in payload:
            payload["refresh_interval_s"] = float(refresh_interval_s)
            changed = True
        params = payload.setdefault("parameters", {})
        for name in ONE_MW2_LINEAL_NAMES:
            if name not in params:
                params[name] = float(defaults[name])
                changed = True
        if changed:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        return path
    payload = {
        "description": "Manual 1MW2 lineal beam model parameters. The quadratic beam term b is fixed to 0.",
        "root_index": int(root_index),
        "refresh_interval_s": float(refresh_interval_s),
        "parameters": {name: float(defaults[name]) for name in ONE_MW2_LINEAL_NAMES},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def read_1mw2_lineal_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    params = default_1mw2_lineal_parameters()
    params.update({key: float(value) for key, value in payload.get("parameters", {}).items() if key in params})
    return {
        "payload": payload,
        "root_index": int(payload.get("root_index", payload.get("metadata", {}).get("root_index", 0))),
        "refresh_interval_s": max(float(payload.get("refresh_interval_s", 2.0)), 0.2),
        "parameters": {name: float(params[name]) for name in ONE_MW2_LINEAL_NAMES},
        "parameter_names": ONE_MW2_LINEAL_NAMES,
    }


def evaluate_1mw2_lineal_model(params, root_file):
    z, charge, charge_raw, raw_norm = load_1mw2_beam_profile(root_file)
    z_grid = np.linspace(float(np.nanmin(z)) - 100.0, float(np.nanmax(z)) + 500.0, 20001)
    z0_step = float(params["z0_step"])
    z2 = float(params["z2"])
    z0_ab = z0_step + float(params["z0_ab_frac"]) * z2
    profile = perfil_carga_sep_z0(
        z,
        z_grid,
        zR0=params["zR0"],
        z0_step=z0_step,
        z2=z2,
        z0_ab=z0_ab,
        a=params["a"],
        b=0.0,
        A_area=params["A_area"],
    )
    model = float(params["y0"]) + float(params["K"]) * profile
    residual = charge - model
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum((charge - np.mean(charge))**2))
    r2 = 1.0 - float(np.sum(residual**2)) / denom if denom > 0 else float("nan")
    data_peak, _, data_fwhm = profile_peak_fwhm(z, charge)
    model_peak, _, model_fwhm = profile_peak_fwhm(z, model)
    return {
        "z": z,
        "charge": charge,
        "charge_raw": charge_raw,
        "raw_norm": raw_norm,
        "z_grid": z_grid,
        "profile": profile,
        "model": model,
        "residual": residual,
        "rmse": rmse,
        "r2": r2,
        "data_peak": data_peak,
        "data_fwhm": data_fwhm,
        "model_peak": model_peak,
        "model_fwhm": model_fwhm,
        "geom": {"z0_step": z0_step, "z0_ab": z0_ab, "z2_end": z0_step + z2, "z2": z2},
    }


def plot_1mw2_lineal_control(root_file, params, json_path):
    result = evaluate_1mw2_lineal_model(params, root_file)
    fig, (ax_profile, ax_resid) = plt.subplots(
        2, 1, figsize=(9.0, 7.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_profile.plot(result["z"], result["charge"], ".", color="black", label="1MW2 data / raw max")
    ax_profile.plot(result["z"], result["model"], "-", color="tab:red", label="lineal beam model")
    ax_profile.axvline(result["geom"]["z0_step"], linestyle="--", color="gray", label="z0_step")
    ax_profile.axvline(result["geom"]["z2_end"], linestyle="--", color="black", label="z0_step + z2")
    ax_profile.axvline(result["geom"]["z0_ab"], linestyle=":", color="tab:green", label="z0_ab")
    ax_profile.set_ylabel("charge / raw max")
    ax_profile.grid(alpha=0.25)
    ax_profile.legend(loc="best", fontsize=8)
    ax_profile.set_title(
        f"1MW2 lineal beam fit: RMSE={result['rmse']:.5g}, R2={result['r2']:.6g}; "
        f"peak {result['model_peak']:.4g}/{result['data_peak']:.4g}, "
        f"FWHM {result['model_fwhm']:.4g}/{result['data_fwhm']:.4g}"
    )
    ax_resid.plot(result["z"], result["residual"], ".-", color="tab:purple", label="data - model")
    ax_resid.axhline(0.0, color="black", lw=1)
    ax_resid.set_xlabel("z (SiC) [um]")
    ax_resid.set_ylabel("residual")
    ax_resid.grid(alpha=0.25)
    ax_resid.legend(loc="best", fontsize=8)
    fig.suptitle(str(json_path), fontsize=10, y=0.995)
    fig.tight_layout()
    return fig, result


def watch_1mw2_lineal_json(
    path,
    beam_json_path=CONFIG_DIR / "1MW2_beam_fit_params.json",
):
    ensure_1mw2_lineal_json(path, beam_json_path=beam_json_path)
    root_records = discover_1mw2_root_files()
    if not root_records:
        raise RuntimeError(f"No 1MW2 ROOT files found in {ROOT_DIR}")

    last_signature = None
    refresh_s = 2.0
    while True:
        try:
            path = Path(path)
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != last_signature:
                config = read_1mw2_lineal_json(path)
                refresh_s = config["refresh_interval_s"]
                root_index = config["root_index"]
                if root_index < 0 or root_index >= len(root_records):
                    raise IndexError(f"root_index={root_index} out of range 0..{len(root_records) - 1}")
                root_file = root_records[root_index]["root_file"]

                clear_output(wait=True)
                print("Manual parameter JSON:", path)
                print("Beam model: 1MW2 lineal, with b fixed to 0")
                print("Refresh interval [s]:", refresh_s)
                print("Root index:", root_index)
                print("Profile root:", root_file)
                for name in config["parameter_names"]:
                    print(f"{name:>12s} = {config['parameters'][name]:.8g}")
                fig, result = plot_1mw2_lineal_control(root_file, config["parameters"], path)
                print(f"RMSE={result['rmse']:.8g}, R2={result['r2']:.8g}")
                print(f"Data/Fit peak: {result['data_peak']:.8g} / {result['model_peak']:.8g}")
                print(f"Data/Fit FWHM: {result['data_fwhm']:.8g} / {result['model_fwhm']:.8g}")
                display(fig)
                plt.close(fig)
                last_signature = signature
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            clear_output(wait=True)
            print("Error while reading/plotting 1MW2 lineal JSON:", exc)
            print("JSON path:", path)
        time.sleep(refresh_s)
