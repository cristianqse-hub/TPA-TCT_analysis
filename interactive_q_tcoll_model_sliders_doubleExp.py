#!/usr/bin/env python3
"""Interactive double-exponential Q+TColl model viewer.

The script exposes EF_*, SC_*, TR_* and RC_* values as Matplotlib sliders and
updates the Q+TColl manual JSON on every slider change. Parameter types,
constraints and limits already present in the JSON are preserved.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

import trapping_model_Q_TColl_li as qtc


DEFAULT_CONFIG = Path("SiC_UVLED/configs/Q_TColl_manual_single_profile_control_doubleExp.json")
DEFAULT_OUTPUT_JSON = DEFAULT_CONFIG
DEFAULT_ROOT_DIR = Path("SiC_UVLED/RootFiles")
PROFILE_SIGMA_FRACTION = 0.0175


SLIDER_DEFAULT_RANGES = {
    "EF_BiasVoltage": (-1500.0, 1500.0),
    "EF_ExpAmpLeft": (0.0, 250.0),
    "EF_ExpDecayLeft": (0.2, 250.0),
    "EF_ExpDecayRight": (0.2, 250.0),
    "SC_scaleAmp": (0.1, 3.0),
    "SC_scaleOffset": (-0.25, 0.25),
    "SC_scale_zShift": (-20.0, 20.0),
    "TR_tau_e": (1e-3, 20.0),
    "TR_tau_h": (1e-3, 20.0),
    "RC_capacitance_pF": (1.0, 100.0),
    "RC_resistance_ohm": (1.0, 500.0),
    "RC_extra_sigma_ns": (0.0, 5.0),
}


SLIDER_NAMES = [
    "profile_index",
    "EF_BiasVoltage",
    "EF_ExpAmpLeft",
    "EF_ExpDecayLeft",
    "EF_ExpDecayRight",
    "SC_scaleAmp",
    "SC_scaleOffset",
    "SC_scale_zShift",
    "TR_tau_e",
    "TR_tau_h",
    "RC_capacitance_pF",
    "RC_resistance_ohm",
    "RC_extra_sigma_ns",
]


def load_payload(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if "configuration" not in payload:
        payload = {"configuration": payload}
    return payload


def load_configuration(config_path: Path) -> dict:
    payload = load_payload(config_path)
    config = qtc.load_configuration(payload["configuration"])
    config["fit_options"]["field_model"] = "double_exponential"
    config["parameters"]["EF_BiasVoltage"]["enabled"] = True
    for name in ("EF_CoefA", "EF_CoefB", "EF_CoefC"):
        if name in config["parameters"]:
            config["parameters"][name].update({"type": "fixed", "value": 0.0})
    if "EF_ExpAmpRight" in config["parameters"]:
        config["parameters"]["EF_ExpAmpRight"]["type"] = "fixed"
    return qtc.load_configuration(config)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def initial_profile_index(config_path: Path, rootfiles: list[str]) -> int:
    try:
        payload = load_payload(config_path)
        index = int(payload.get("profile_index", 4))
    except Exception:
        index = 4
    return max(0, min(index, len(rootfiles) - 1))


def slider_limits(name: str, spec: dict) -> tuple[float, float]:
    value = float(spec.get("value", 0.0))
    if "min" in spec and "max" in spec:
        low = float(spec["min"])
        high = float(spec["max"])
    else:
        low, high = SLIDER_DEFAULT_RANGES.get(name, (value - 1.0, value + 1.0))
    if low <= value <= high:
        return low, high
    margin = max(abs(value), 1.0)
    return min(low, value - margin), max(high, value + margin)


def configuration_from_sliders(base_config: dict, sliders: dict[str, Slider]) -> dict:
    config = copy.deepcopy(base_config)
    for name in SLIDER_NAMES:
        if name == "profile_index" or name not in config["parameters"]:
            continue
        config["parameters"][name]["value"] = float(sliders[name].val)
    config["fit_options"]["field_model"] = "double_exponential"
    config["parameters"]["EF_BiasVoltage"]["enabled"] = True
    return qtc.load_configuration(config)


def load_selected_data(profile_index: int, config: dict, rootfiles: list[str]) -> dict:
    error_options = qtc.data_error_options(config)
    selection_options = qtc.data_selection_options(config)
    options = config["fit_options"]
    x_q, y_q, y_q_err, rootfilename, q_error_source = qtc.get_profile_data_with_errors(
        profile_index,
        rootfiles,
        normalize_to_1mw2=True,
        absolute_error=error_options["charge"]["absolute"],
        systematic_error=error_options["charge"]["systematic"],
        fallback_fraction=float(options.get("charge_error_fraction", PROFILE_SIGMA_FRACTION)),
    )
    (x_q_used, y_q_used, y_q_err_used), q_masked, _ = qtc.split_data_by_index_selection(
        x_q, y_q, y_q_err, **selection_options["charge"]
    )
    data = {
        "x_q": x_q_used,
        "y_q": y_q_used,
        "y_q_err": y_q_err_used,
        "rootfilename": rootfilename,
        "q_error_source": q_error_source,
        "masked_charge_data": q_masked,
        "tcoll_threshold": float(options.get("tColl_threshold_percent", 5.0)),
    }
    try:
        x_t, y_t, y_t_err, _, threshold = qtc.get_tcoll_data(
            profile_index,
            rootfiles,
            threshold_percent=data["tcoll_threshold"],
            source_tree=options.get("tColl_source_tree", "SignalTPA_best_masked"),
            absolute_error_ns=error_options["tColl"]["absolute_ns"],
            systematic_error_ns=error_options["tColl"]["systematic_ns"],
        )
        (x_t_used, y_t_used, y_t_err_used), t_masked, _ = qtc.split_data_by_index_selection(
            x_t, y_t, y_t_err, **selection_options["tColl"]
        )
        data.update({
            "x_t": x_t_used,
            "y_t": y_t_used,
            "y_t_err": y_t_err_used,
            "masked_tcoll_data": t_masked,
            "tcoll_threshold": threshold,
        })
    except Exception:
        data.update({"x_t": None, "y_t": None, "y_t_err": None, "masked_tcoll_data": None})
    return data


def evaluate_case(base_config, sliders, rootfiles):
    profile_index = int(round(sliders["profile_index"].val))
    profile_index = max(0, min(profile_index, len(rootfiles) - 1))
    config = configuration_from_sliders(base_config, sliders)
    data = load_selected_data(profile_index, config, rootfiles)
    model = qtc.simulate_q_tcoll_model(data["x_q"], config)
    plot_data = {"x_q": data["x_q"], "y_q": data["y_q"], "y_q_err": data["y_q_err"]}
    if data["x_t"] is not None:
        plot_data.update({"x_t": data["x_t"], "y_t": data["y_t"], "y_t_err": data["y_t_err"]})
    q_masked = data["masked_charge_data"]
    if q_masked is not None and q_masked[0].size:
        plot_data.update({"x_q_masked": q_masked[0], "y_q_masked": q_masked[1], "y_q_masked_err": q_masked[2]})
    t_masked = data["masked_tcoll_data"]
    if t_masked is not None and t_masked[0].size:
        plot_data.update({"x_t_masked": t_masked[0], "y_t_masked": t_masked[1], "y_t_masked_err": t_masked[2]})
    metrics = qtc.compute_q_tcoll_metrics(model, data=plot_data)
    return {
        "profile_index": profile_index,
        "rootfilename": data["rootfilename"],
        "configuration": config,
        "parameters": qtc.parameter_values(config),
        "model": model,
        "data": plot_data,
        "metrics": metrics,
        "q_error_source": data["q_error_source"],
    }


def save_control_json(fallback_template_payload, result, output_json: Path) -> None:
    if output_json.exists():
        payload = load_payload(output_json)
    else:
        payload = copy.deepcopy(fallback_template_payload)
    payload["description"] = (
        "Interactive Q+TColl double-exponential profile control. Updated by "
        "interactive_q_tcoll_model_sliders_doubleExp.py whenever sliders change."
    )
    payload["profile_index"] = int(result["profile_index"])
    payload["rootfilename"] = str(result["rootfilename"])
    payload.setdefault("configuration", {})
    payload["configuration"].setdefault("parameters", {})
    payload["configuration"].setdefault("fit_options", {})
    payload["configuration"]["fit_options"]["field_model"] = "double_exponential"
    for name in SLIDER_NAMES:
        if name == "profile_index":
            continue
        if name in payload["configuration"]["parameters"] and name in result["parameters"]:
            payload["configuration"]["parameters"][name]["value"] = result["parameters"][name]
    payload["interactive_state"] = {
        "source_script": "interactive_q_tcoll_model_sliders_doubleExp.py",
        "field_model": "double_exponential",
        "generation_method": result["configuration"]["fit_options"].get("generation_method"),
        "chi2_charge_dof": result["metrics"].get("charge", {}).get("chi2_dof"),
        "chi2_tcoll_dof": result["metrics"].get("tcoll", {}).get("chi2_dof"),
        "chi2_combined_dof": result["metrics"].get("combined", {}).get("chi2_dof"),
        "updated_parameters": {
            name: result["parameters"][name]
            for name in SLIDER_NAMES
            if name != "profile_index" and name in result["parameters"]
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(json_ready(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def safe_plot(axis, x, y, *args, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape == y.shape and x.size:
        axis.plot(x, y, *args, **kwargs)


def safe_errorbar(axis, x, y, yerr=None, *args, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape == y.shape and x.size:
        axis.errorbar(x, y, yerr=yerr, *args, **kwargs)


def draw_diagnostics(fig, axes, result):
    ax_q, ax_q_res, ax_t, ax_t_res, ax_field, ax_v, ax_resp, ax_map = axes
    for axis in axes:
        axis.clear()

    model = result["model"]
    data = result["data"]
    metrics = result["metrics"]
    x = np.asarray(model["x"], dtype=float)
    offset = float(model.get("charge_offset", 0.0))
    q_model = np.asarray(model["charge_total"], dtype=float) - offset
    q_e = np.asarray(model["charge_e"], dtype=float)
    q_h = np.asarray(model["charge_h"], dtype=float)
    x_q = np.asarray(data["x_q"], dtype=float)
    y_q = np.asarray(data["y_q"], dtype=float) - offset
    y_q_err = np.asarray(data["y_q_err"], dtype=float)

    safe_errorbar(ax_q, x_q, y_q, y_q_err, fmt="o", ms=3, color="black", ecolor="0.65", label="data")
    if "x_q_masked" in data:
        safe_errorbar(
            ax_q,
            data["x_q_masked"],
            np.asarray(data["y_q_masked"], dtype=float) - offset,
            data["y_q_masked_err"],
            fmt="o",
            ms=3,
            color="red",
            ecolor="red",
            alpha=0.7,
            label="masked",
        )
    q_chi = metrics.get("charge", {}).get("chi2_dof", np.nan)
    safe_plot(ax_q, x, q_model, color="tab:orange", lw=2, label=f"fit, chi2/N={q_chi:.4g}")
    safe_plot(ax_q, x, q_e, color="darkblue", lw=1.3, alpha=0.8, label="e")
    safe_plot(ax_q, x, q_h, color="crimson", lw=1.3, alpha=0.8, label="h")
    ax_q.set(ylabel="charge [NE]", title="charge profile")
    ax_q.legend(frameon=False, fontsize=8)

    q_interp = np.interp(x_q, x, q_model, left=np.nan, right=np.nan)
    safe_errorbar(ax_q_res, x_q, y_q - q_interp, y_q_err, fmt="o", ms=3, color="black", ecolor="0.65")
    ax_q_res.axhline(0.0, color="black", lw=1)
    ax_q_res.set(xlabel="z (SiC) [um]", ylabel="data - fit [NE]", title="charge residual")

    tcoll = np.asarray(model["tcoll_rc"], dtype=float)
    threshold = model.get("threshold_percent", result["configuration"]["fit_options"].get("tColl_threshold_percent", np.nan))
    if "x_t" in data:
        safe_errorbar(ax_t, data["x_t"], data["y_t"], data["y_t_err"], fmt="o", ms=3, color="black", ecolor="0.65", label="data")
    if "x_t_masked" in data:
        safe_errorbar(ax_t, data["x_t_masked"], data["y_t_masked"], data["y_t_masked_err"], fmt="o", ms=3, color="red", ecolor="red", alpha=0.7, label="masked")
    t_chi = metrics.get("tcoll", {}).get("chi2_dof", np.nan)
    safe_plot(ax_t, x, tcoll, color="purple", lw=2, label=f"fit, chi2/N={t_chi:.4g}")
    ax_t.set(ylabel="duration [ns]", title=f"pulse-duration profile (th={threshold:g}%)")
    ax_t.legend(frameon=False, fontsize=8)
    if "x_t" in data:
        t_interp = np.interp(data["x_t"], x, tcoll, left=np.nan, right=np.nan)
        safe_errorbar(ax_t_res, data["x_t"], np.asarray(data["y_t"]) - t_interp, data["y_t_err"], fmt="o", ms=3, color="black", ecolor="0.65")
    ax_t_res.axhline(0.0, color="black", lw=1)
    ax_t_res.set(xlabel="z (SiC) [um]", ylabel="data - fit [ns]", title="duration residual")

    response = model["response"]
    z = np.asarray(response.get("z", []), dtype=float)
    safe_plot(ax_field, z, response.get("efield", []), color="tab:green")
    ax_field.axhline(1e-2, color="tab:red", ls="--", lw=0.8)
    ax_field.set(title="electric field", xlabel="z (SiC) [um]", ylabel="field [V/um]")

    safe_plot(ax_v, z, response.get("vdrift_e", []), color="darkblue", label="e")
    safe_plot(ax_v, z, response.get("vdrift_h", []), color="crimson", label="h")
    ax_v.set_yscale("log")
    ax_v.set(title="drift velocity", xlabel="z (SiC) [um]", ylabel="velocity [um/ns]")
    if ax_v.get_legend_handles_labels()[0]:
        ax_v.legend(frameon=False, fontsize=8)

    safe_plot(ax_resp, z, response.get("response_total", []), color="black", label="total")
    safe_plot(ax_resp, z, response.get("response_e", []), color="darkblue", alpha=0.8, label="e")
    safe_plot(ax_resp, z, response.get("response_h", []), color="crimson", alpha=0.8, label="h")
    ax_resp.set(title="CCE", xlabel="z (SiC) [um]", ylabel="CCE")
    if ax_resp.get_legend_handles_labels()[0]:
        ax_resp.legend(frameon=False, fontsize=8)

    safe_plot(ax_map, z, response.get("pulse_duration_intrinsic", []), color="black", label="total")
    safe_plot(ax_map, z, response.get("pulse_duration_e_intrinsic", []), color="darkblue", alpha=0.8, label="e")
    safe_plot(ax_map, z, response.get("pulse_duration_h_intrinsic", []), color="crimson", alpha=0.8, label="h")
    ax_map.set(title="collection time", xlabel="z (SiC) [um]", ylabel="duration [ns]")
    if ax_map.get_legend_handles_labels()[0]:
        ax_map.legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.25)
    root_name = Path(result["rootfilename"]).name
    combined_chi = metrics.get("combined", {}).get("chi2_dof", np.nan)
    fig.suptitle(
        f"{root_name} | combined chi2/N={combined_chi:.4g} | "
        f"method={result['configuration']['fit_options'].get('generation_method')}",
        fontsize=11,
    )
    fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="Interactive Q+TColl double-exponential model sliders.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    args = parser.parse_args()

    rootfiles = qtc.discover_profile_rootfiles(args.root_dir)
    if not rootfiles:
        raise RuntimeError(f"No ROOT files found under {args.root_dir}")
    template_payload = load_payload(args.config)
    base_config = load_configuration(args.config)
    profile_index0 = initial_profile_index(args.config, rootfiles)

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(4, 4, left=0.06, right=0.98, top=0.89, bottom=0.32, hspace=0.55, wspace=0.38)
    axes = [
        fig.add_subplot(grid[0:2, 0:2]),
        fig.add_subplot(grid[2, 0:2]),
        fig.add_subplot(grid[0:2, 2:4]),
        fig.add_subplot(grid[2, 2:4]),
        fig.add_subplot(grid[3, 0]),
        fig.add_subplot(grid[3, 1]),
        fig.add_subplot(grid[3, 2]),
        fig.add_subplot(grid[3, 3]),
    ]

    sliders = {}
    slider_height = 0.016
    slider_gap = 0.022
    slider_bottom = 0.025
    for idx, name in enumerate(SLIDER_NAMES):
        ypos = slider_bottom + (len(SLIDER_NAMES) - 1 - idx) * slider_gap
        axis = fig.add_axes([0.20, ypos, 0.58, slider_height])
        if name == "profile_index":
            slider = Slider(axis, name, 0, len(rootfiles) - 1, valinit=profile_index0, valstep=1, valfmt="%0.0f", dragging=False)
        else:
            spec = base_config["parameters"][name]
            low, high = slider_limits(name, spec)
            slider = Slider(axis, name, low, high, valinit=float(spec["value"]), valfmt="%.6g", dragging=False)
        sliders[name] = slider

    reset_axis = fig.add_axes([0.82, 0.025, 0.08, 0.032])
    reset_button = Button(reset_axis, "Reset")
    status_axis = fig.add_axes([0.80, 0.07, 0.18, 0.22])
    status_axis.axis("off")
    status_text = status_axis.text(0.0, 1.0, "", va="top", family="monospace", fontsize=8)

    def update(_=None):
        try:
            current_config_path = args.output_json if args.output_json.exists() else args.config
            current_config = load_configuration(current_config_path)
            result = evaluate_case(current_config, sliders, rootfiles)
            save_control_json(template_payload, result, args.output_json)
            draw_diagnostics(fig, axes, result)
            params = result["parameters"]
            status_text.set_text(
                "Current values\n"
                f"EF_ExpAmpLeft={params['EF_ExpAmpLeft']:.6g}\n"
                f"EF_ExpDecayLeft={params['EF_ExpDecayLeft']:.6g}\n"
                f"EF_ExpAmpRight={params['EF_ExpAmpRight']:.6g}\n"
                f"EF_ExpDecayRight={params['EF_ExpDecayRight']:.6g}\n"
                f"TR_tau_e={params['TR_tau_e']:.6g}\n"
                f"TR_tau_h={params['TR_tau_h']:.6g}\n"
                f"RC_tau={qtc.rc_tau_ns_from_values(params):.6g} ns\n"
                f"Q err={result['q_error_source']}\n"
                f"saved={args.output_json}"
            )
        except Exception as exc:
            status_text.set_text(f"Evaluation failed:\n{exc}")
            fig.canvas.draw_idle()

    def reset(_=None):
        for slider in sliders.values():
            slider.reset()

    for slider in sliders.values():
        slider.on_changed(update)
    reset_button.on_clicked(reset)

    print(f"Loaded config: {args.config}")
    print(f"Loaded {len(rootfiles)} ROOT profiles from {args.root_dir}")
    print(f"Updating JSON on every change: {args.output_json}")
    update()
    plt.show()


if __name__ == "__main__":
    main()
