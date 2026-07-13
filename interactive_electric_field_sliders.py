#!/usr/bin/env python3
"""Interactive electric-field viewer with Matplotlib sliders.

The parameter names match the RooFit/trapping config names. The field follows
the same voltage constraint used by trapping_model_RooFit_li.py:

    E(z) = EF_CoefA * u**2 + EF_CoefB * u + EF_CoefC

with u measured inside the active region relative to EF_z0, and EF_CoefC
derived so that integral(E dz) over BM_zRight equals EF_BiasVoltage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider


DEFAULT_CONFIG = Path("SiC_UVLED/configs/RooFit_manual_single_profile_control.json")
FIELD_FLOOR = 1e-2


DEFAULT_VALUES = {
    "BM_z0": -2.561761988852449,
    "BM_zRight": 50.337266961172645,
    "SC_scale_zShift": 0.0,
    "EF_BiasVoltage": 850.0,
    "EF_CoefA": 0.0,
    "EF_CoefB": 0.0,
    "EF_CoefC": 0.0,
    "EF_z0": 0.0,
}


def load_config_values(config_path: Path | None) -> dict[str, float]:
    values = dict(DEFAULT_VALUES)
    if config_path is None or not config_path.exists():
        return values

    with config_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    config = payload.get("configuration", payload)
    parameters = config.get("parameters", {})
    for name in values:
        spec = parameters.get(name)
        if isinstance(spec, dict) and "value" in spec:
            values[name] = float(spec["value"])
    return values


def slider_limits(name: str, value: float) -> tuple[float, float]:
    ranges = {
        "BM_z0": (-20.0, 20.0),
        "BM_zRight": (1.0, 120.0),
        "SC_scale_zShift": (-20.0, 20.0),
        "EF_BiasVoltage": (-1500.0, 1500.0),
        "EF_CoefA": (-0.1, 0.1),
        "EF_CoefB": (-1.0, 1.0),
        "EF_z0": (-80.0, 80.0),
    }
    if name in ranges:
        low, high = ranges[name]
    else:
        scale = max(abs(value), 1.0)
        low, high = value - 5.0 * scale, value + 5.0 * scale
    if low <= value <= high:
        return low, high
    margin = max(abs(value), 1.0)
    return min(low, value - margin), max(high, value + margin)


def derived_EF_CoefC(values: dict[str, float]) -> float:
    width = float(values["BM_zRight"])
    if not np.isfinite(width) or width == 0.0:
        return np.nan
    center = float(values["EF_z0"])
    u_left = -center
    u_right = width - center
    quadratic_integral = float(values["EF_CoefA"]) * (u_right**3 - u_left**3) / 3.0
    linear_integral = float(values["EF_CoefB"]) * (u_right**2 - u_left**2) / 2.0
    return (float(values["EF_BiasVoltage"]) - quadratic_integral - linear_integral) / width


def evaluate_field(values: dict[str, float], n_points: int = 800):
    width = max(float(values["BM_zRight"]), 1e-12)
    z_start = float(values["BM_z0"]) + float(values["SC_scale_zShift"])
    active_z = np.linspace(0.0, width, n_points)
    z_abs = z_start + active_z
    u = active_z - float(values["EF_z0"])
    coef_c = derived_EF_CoefC(values)
    e_raw = float(values["EF_CoefA"]) * u * u + float(values["EF_CoefB"]) * u + coef_c
    e_physical = np.maximum(e_raw, 0.0)
    voltage_raw = np.trapz(e_raw, active_z)
    voltage_physical = np.trapz(e_physical, active_z)
    return z_abs, active_z, e_raw, e_physical, coef_c, voltage_raw, voltage_physical


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive electric-field sliders.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config JSON to read initial values from. Default: {DEFAULT_CONFIG}",
    )
    args = parser.parse_args()

    values = load_config_values(args.config)
    slider_names = [
        "EF_BiasVoltage",
        "EF_CoefA",
        "EF_CoefB",
        "EF_z0",
        "BM_z0",
        "BM_zRight",
        "SC_scale_zShift",
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    plt.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.42)

    z_abs, active_z, e_raw, e_phys, coef_c, v_raw, v_phys = evaluate_field(values)
    raw_line, = ax.plot(z_abs, e_raw, color="tab:blue", lw=1.8, label="raw polynomial E(z)")
    physical_line, = ax.plot(z_abs, e_phys, color="black", lw=2.0, label="physical max(E, 0)")
    floor_line = ax.axhline(FIELD_FLOOR, color="tab:red", lw=1.0, ls="--", label="1e-2 V/um")
    zero_line = ax.axhline(0.0, color="0.4", lw=0.8)
    low_field_fill = ax.fill_between(
        z_abs,
        0.0,
        e_phys,
        where=e_phys < FIELD_FLOOR,
        color="tab:red",
        alpha=0.18,
        label="E < 1e-2 V/um",
    )

    ax.set_xlabel("z / um")
    ax.set_ylabel("electric field / (V/um)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    title = ax.set_title("")

    sliders = {}
    slider_height = 0.026
    slider_gap = 0.043
    slider_bottom = 0.08
    for idx, name in enumerate(slider_names):
        ypos = slider_bottom + (len(slider_names) - 1 - idx) * slider_gap
        slider_ax = fig.add_axes([0.19, ypos, 0.68, slider_height])
        low, high = slider_limits(name, values[name])
        slider = Slider(
            ax=slider_ax,
            label=name,
            valmin=low,
            valmax=high,
            valinit=values[name],
            valfmt="%.6g",
        )
        sliders[name] = slider

    check_ax = fig.add_axes([0.01, 0.73, 0.13, 0.10])
    check = CheckButtons(check_ax, ["raw", "physical"], [True, True])

    reset_ax = fig.add_axes([0.88, 0.08, 0.09, 0.04])
    reset_button = Button(reset_ax, "Reset")

    def current_values() -> dict[str, float]:
        current = dict(values)
        for key, slider in sliders.items():
            current[key] = float(slider.val)
        return current

    def update(_=None) -> None:
        nonlocal low_field_fill
        current = current_values()
        z_abs_new, _, e_raw_new, e_phys_new, coef_c_new, v_raw_new, v_phys_new = evaluate_field(current)
        raw_line.set_data(z_abs_new, e_raw_new)
        physical_line.set_data(z_abs_new, e_phys_new)
        low_field_fill.remove()
        low_field_fill = ax.fill_between(
            z_abs_new,
            0.0,
            e_phys_new,
            where=e_phys_new < FIELD_FLOOR,
            color="tab:red",
            alpha=0.18,
        )
        xmin, xmax = np.nanmin(z_abs_new), np.nanmax(z_abs_new)
        ymax = max(float(np.nanmax(e_raw_new)), float(np.nanmax(e_phys_new)), FIELD_FLOOR)
        ymin = min(float(np.nanmin(e_raw_new)), 0.0)
        yrange = max(ymax - ymin, 1.0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin - 0.08 * yrange, ymax + 0.12 * yrange)
        title.set_text(
            "EF_CoefC = "
            f"{coef_c_new:.6g} V/um | integral raw = {v_raw_new:.6g} V | "
            f"integral max(E,0) = {v_phys_new:.6g} V"
        )
        fig.canvas.draw_idle()

    def toggle_visibility(label: str) -> None:
        if label == "raw":
            raw_line.set_visible(not raw_line.get_visible())
        elif label == "physical":
            physical_line.set_visible(not physical_line.get_visible())
        fig.canvas.draw_idle()

    def reset(_=None) -> None:
        for name, slider in sliders.items():
            slider.reset()

    for slider in sliders.values():
        slider.on_changed(update)
    check.on_clicked(toggle_visibility)
    reset_button.on_clicked(reset)

    update()
    print(f"Loaded initial values from: {args.config}")
    plt.show()


if __name__ == "__main__":
    main()
