import json
import copy
import contextlib
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

import trapping_model_li as _base


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

MODEL_PARAMETER_NAMES = _base.MODEL_PARAMETER_NAMES
PARAMETER_UNITS = _base.PARAMETER_UNITS
DEFAULT_N_Z_GRID = _base.DEFAULT_N_Z_GRID


def parameter_label(name):
    return _base.parameter_label(name)


def discover_profile_rootfiles(root_dir="SiC_UVLED/RootFiles"):
    return sorted(str(path) for path in Path(root_dir).glob("*.root"))


def _profile_charge_arrays(rootfilename):
    from utils_lib import getVals

    vals = getVals(rootfilename, ["Profiles:zSiC", "Profiles:ChargeCSPA_Avr"])
    xdata = np.asarray(vals["Profiles:zSiC"], dtype=float).ravel()
    ydata = np.abs(np.asarray(vals["Profiles:ChargeCSPA_Avr"], dtype=float).ravel())
    n_common = min(xdata.size, ydata.size)
    xdata = xdata[:n_common]
    ydata = ydata[:n_common]
    finite = np.isfinite(xdata) & np.isfinite(ydata)
    xdata = xdata[finite]
    ydata = ydata[finite]
    order = np.argsort(xdata)
    return xdata[order], ydata[order]


def _profile_normalization_from_1mw2(rootfiles_list):
    rootfiles = list(rootfiles_list)
    matches = [rootfile for rootfile in rootfiles if "1MW2" in Path(rootfile).name]
    if not matches:
        raise FileNotFoundError("No 1MW2 ROOT file found in rootfiles_list for profile normalization")
    _, y_1mw2 = _profile_charge_arrays(matches[0])
    norm = float(np.nanmax(np.abs(y_1mw2)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid 1MW2 normalization from {matches[0]}")
    return norm


def get_profileData(index, rootfiles_list):
    rootfiles = list(rootfiles_list)
    if not rootfiles:
        raise ValueError("rootfiles_list is empty")
    index = int(index)
    if index < 0 or index >= len(rootfiles):
        raise IndexError(f"index={index} out of range 0..{len(rootfiles) - 1}")
    rootfilename = str(rootfiles[index])
    xdata, ydata = _profile_charge_arrays(rootfilename)
    norm = _profile_normalization_from_1mw2(rootfiles)
    return xdata, ydata / norm, rootfilename


def load_fit_configuration(configuration):
    raw_parameters = {}
    if isinstance(configuration, dict):
        raw_parameters = copy.deepcopy(configuration.get("parameters", {}))
    config = _base.load_fit_configuration(configuration)
    config["parameters"]["EF_CoefC"].update({"type": "fixed", "value": 0.0})
    config["parameters"]["EF_BiasVoltage"]["enabled"] = True
    for key in ("min", "max", "constraint", "mean", "central", "sigma"):
        config["parameters"]["EF_CoefC"].pop(key, None)
    for name, spec in config["parameters"].items():
        raw_spec = raw_parameters.get(name, {})
        constraint = spec.get("constraint", raw_spec.get("constraint"))
        if constraint is None:
            if "mean" in raw_spec and "sigma" in raw_spec:
                constraint = {"mean": raw_spec["mean"], "sigma": raw_spec["sigma"]}
            elif "central" in raw_spec and "sigma" in raw_spec:
                constraint = {"mean": raw_spec["central"], "sigma": raw_spec["sigma"]}
            elif "mean" in spec and "sigma" in spec:
                constraint = {"mean": spec["mean"], "sigma": spec["sigma"]}
            elif "central" in spec and "sigma" in spec:
                constraint = {"mean": spec["central"], "sigma": spec["sigma"]}
        if constraint is not None:
            mean = float(constraint["mean"])
            sigma = float(constraint["sigma"])
            if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"Gaussian constraint for '{name}' requires finite mean and sigma > 0")
            spec["constraint"] = {"mean": mean, "sigma": sigma}
    options = config.setdefault("fit_options", {})
    options.setdefault("relative_sigma_fraction", 0.02)
    options.setdefault("relative_sigma_floor_fraction", 1e-6)
    options.setdefault("minimizer_type", "Minuit2")
    options.setdefault("minimizer_algorithm", "Minimize")
    options.setdefault("strategy", 1)
    options.setdefault("print_level", -1)
    options.setdefault("constraint_range_sigmas", 10.0)
    options.setdefault("do_useMINOS", False)
    options.setdefault("params_MINOS", [])
    options.setdefault("table_sigfigs", 4)
    options.setdefault("refit_fromLocalMinimumMINOS", False)
    options.setdefault("show_roofit_output", False)
    options.setdefault("field_model", "polynomial")
    options["field_model"] = _canonical_field_model(options["field_model"])
    return config


def _canonical_field_model(field_model):
    text = str(field_model).strip().lower().replace("-", "_")
    aliases = {
        "poly": "polynomial",
        "polynomial": "polynomial",
        "double_exp": "double_exponential",
        "double_exponential": "double_exponential",
        "doublejunction": "double_exponential",
        "double_junction": "double_exponential",
    }
    if text not in aliases:
        raise ValueError(
            "fit_options['field_model'] must be 'polynomial' or 'double_exponential'"
        )
    return aliases[text]


def _field_model_kind(config):
    return 1 if _canonical_field_model(config["fit_options"].get("field_model", "polynomial")) == "double_exponential" else 0


def _std_vector(values):
    import ROOT

    vector = ROOT.std.vector("double")()
    for value in np.asarray(values, dtype=float).ravel():
        vector.push_back(float(value))
    return vector


@contextlib.contextmanager
def _suppress_c_output(enabled):
    if not enabled:
        yield
        return
    stdout_fd = 1
    stderr_fd = 2
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _compile_cpp_model():
    import ROOT

    if getattr(ROOT, "TrappingGaussianNLLV5", None) is not None:
        return ROOT

    ROOT.gInterpreter.Declare(
        r'''
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooListProxy.h"

namespace trapping_roofit_v5 {

enum ParameterIndex {
    BM_z0 = 0, BM_zRight, BM_zR0, BM_z_Aberr,
    BM_CoefA, BM_CoefB, BM_area, BM_scaleAmp, BM_scaleOffset,
    MV_beta_e, MV_vsat_e, MV_mu0_e,
    MV_beta_h, MV_vsat_h, MV_mu0_h,
    EF_BiasVoltage, EF_CoefA, EF_CoefB, EF_CoefC, EF_z0,
    EF_ExpAmpLeft, EF_ExpDecayLeft, EF_ExpAmpRight, EF_ExpDecayRight,
    SC_scaleAmp, TR_tau_e, TR_tau_h, SC_scaleOffset, SC_scale_zShift,
    N_PARAMETERS
};

struct ProfileResult {
    std::vector<double> total;
    std::vector<double> electron;
    std::vector<double> hole;
    std::vector<double> no_offset;
    double offset = 0.0;
    std::vector<double> z;
    std::vector<double> efield;
    std::vector<double> mobility_e;
    std::vector<double> mobility_h;
    std::vector<double> vdrift_mue;
    std::vector<double> vdrift_muh;
    std::vector<double> response_e;
    std::vector<double> response_h;
    std::vector<double> response_total;
};

inline double trapezoid(const std::vector<double>& y, double dx)
{
    const std::size_t n = y.size();
    if (n < 2) return 0.0;
    double sum = 0.5 * (y.front() + y.back());
    for (std::size_t i = 1; i + 1 < n; ++i) sum += y[i];
    return sum * dx;
}

inline double zR_aberracion_esferica(double z, double zR0, double z0, double a, double b)
{
    const double dz = std::max(z - z0, 0.0);
    return std::max(zR0 + a * dz + b * dz * dz, 1e-12);
}

inline double interp_uniform(
    const std::vector<double>& y, double z_min, double dz, double z
) {
    if (z <= z_min) return y.front();
    const double position = (z - z_min) / dz;
    const auto index = static_cast<std::size_t>(std::floor(position));
    if (index + 1 >= y.size()) return y.back();
    const double fraction = position - static_cast<double>(index);
    return y[index] * (1.0 - fraction) + y[index + 1] * fraction;
}

ProfileResult evaluate_profile(
    const std::vector<double>& x_values,
    const std::vector<double>& p,
    int steps_per_active_region,
    int n_z_grid,
    bool voltage_enabled,
    int field_model_kind
) {
    ProfileResult out;
    if (p.size() < N_PARAMETERS || x_values.empty()) return out;

    const int steps = std::max(steps_per_active_region, 2);
    const int nz = std::max(n_z_grid, 2);
    const double z0 = p[BM_z0];
    const double z2 = p[BM_zRight];
    const double shift = p[SC_scale_zShift];
    const double z0_shifted = z0 + shift;
    const double width = z2;

    out.z.resize(steps);
    const double dz_active = width / static_cast<double>(steps - 1);
    for (int i = 0; i < steps; ++i) {
        out.z[i] = z0_shifted + dz_active * static_cast<double>(i);
    }

    double field_c = p[EF_CoefC];
    double exp_amp_right = p[EF_ExpAmpRight];
    if (field_model_kind == 0 && voltage_enabled) {
        const double center = p[EF_z0];
        const double u_left = -center;
        const double u_right = width - center;
        const double quadratic_integral = p[EF_CoefA] * (std::pow(u_right, 3) - std::pow(u_left, 3)) / 3.0;
        const double linear_integral = p[EF_CoefB] * (u_right * u_right - u_left * u_left) / 2.0;
        field_c = (p[EF_BiasVoltage] - quadratic_integral - linear_integral) / width;
    } else if (field_model_kind == 1 && voltage_enabled) {
        const double lambda_left = std::max(p[EF_ExpDecayLeft], 1e-12);
        const double lambda_right = std::max(p[EF_ExpDecayRight], 1e-12);
        const double left_integral = lambda_left * (1.0 - std::exp(-width / lambda_left));
        const double right_integral = lambda_right * (1.0 - std::exp(-width / lambda_right));
        if (!std::isfinite(left_integral) || !std::isfinite(right_integral) || right_integral <= 0.0) {
            out.total.assign(x_values.size(), std::numeric_limits<double>::quiet_NaN());
            return out;
        }
        exp_amp_right = (p[EF_BiasVoltage] - p[EF_ExpAmpLeft] * left_integral) / right_integral;
    }

    out.efield.resize(steps);
    out.mobility_e.resize(steps);
    out.mobility_h.resize(steps);
    out.vdrift_mue.resize(steps);
    out.vdrift_muh.resize(steps);
    const double field_center_abs = z0_shifted + p[EF_z0];
    const double field_floor = 1e-2;
    for (int i = 0; i < steps; ++i) {
        double efield = 0.0;
        if (field_model_kind == 1) {
            const double s = out.z[i] - z0_shifted;
            const double lambda_left = std::max(p[EF_ExpDecayLeft], 1e-12);
            const double lambda_right = std::max(p[EF_ExpDecayRight], 1e-12);
            efield = (
                p[EF_ExpAmpLeft] * std::exp(-s / lambda_left)
                + exp_amp_right * std::exp(-(width - s) / lambda_right)
            );
        } else {
            const double u = out.z[i] - field_center_abs;
            efield = p[EF_CoefA] * u * u + p[EF_CoefB] * u + field_c;
        }
        if (!std::isfinite(efield)) {
            out.total.assign(x_values.size(), std::numeric_limits<double>::quiet_NaN());
            return out;
        }
        const double drift_field = std::max(efield, 0.0);
        out.efield[i] = drift_field;
        if (drift_field < field_floor) {
            out.mobility_e[i] = 0.0;
            out.mobility_h[i] = 0.0;
            out.vdrift_mue[i] = 0.0;
            out.vdrift_muh[i] = 0.0;
            continue;
        }
        const double mu0_e = p[MV_mu0_e];
        const double mu0_h = p[MV_mu0_h];
        out.mobility_e[i] = mu0_e / std::pow(
            1.0 + std::pow(mu0_e * drift_field / p[MV_vsat_e], p[MV_beta_e]),
            1.0 / p[MV_beta_e]
        );
        out.mobility_h[i] = mu0_h / std::pow(
            1.0 + std::pow(mu0_h * drift_field / p[MV_vsat_h], p[MV_beta_h]),
            1.0 / p[MV_beta_h]
        );
        out.vdrift_mue[i] = out.mobility_e[i] * drift_field;
        out.vdrift_muh[i] = out.mobility_h[i] * drift_field;
    }

    std::vector<double> survival_e(steps - 1), survival_h(steps - 1);
    std::vector<double> response_step_e(steps - 1), response_step_h(steps - 1);
    for (int i = 0; i + 1 < steps; ++i) {
        const bool active_e = (
            out.vdrift_mue[i] > 0.0 && out.vdrift_mue[i + 1] > 0.0
            && std::isfinite(out.vdrift_mue[i]) && std::isfinite(out.vdrift_mue[i + 1])
        );
        const bool active_h = (
            out.vdrift_muh[i] > 0.0 && out.vdrift_muh[i + 1] > 0.0
            && std::isfinite(out.vdrift_muh[i]) && std::isfinite(out.vdrift_muh[i + 1])
        );
        if (active_e && std::isfinite(p[TR_tau_e]) && p[TR_tau_e] > 0.0) {
            const double inverse_velocity = 0.5 * (1.0 / out.vdrift_mue[i] + 1.0 / out.vdrift_mue[i + 1]);
            survival_e[i] = std::exp(-(dz_active * inverse_velocity) / p[TR_tau_e]);
            response_step_e[i] = dz_active / width;
        } else {
            survival_e[i] = 0.0;
            response_step_e[i] = 0.0;
        }
        if (active_h && std::isfinite(p[TR_tau_h]) && p[TR_tau_h] > 0.0) {
            const double inverse_velocity = 0.5 * (1.0 / out.vdrift_muh[i] + 1.0 / out.vdrift_muh[i + 1]);
            survival_h[i] = std::exp(-(dz_active * inverse_velocity) / p[TR_tau_h]);
            response_step_h[i] = dz_active / width;
        } else {
            survival_h[i] = 0.0;
            response_step_h[i] = 0.0;
        }
    }

    out.response_e.assign(steps, 0.0);
    out.response_h.assign(steps, 0.0);
    for (int i = steps - 2; i >= 0; --i) {
        out.response_e[i] = response_step_e[i] + survival_e[i] * out.response_e[i + 1];
    }
    for (int i = 1; i < steps; ++i) {
        out.response_h[i] = response_step_h[i - 1] + survival_h[i - 1] * out.response_h[i - 1];
    }
    out.response_total.resize(steps);
    for (int i = 0; i < steps; ++i) {
        out.response_total[i] = out.response_e[i] + out.response_h[i];
    }

    const auto xminmax = std::minmax_element(x_values.begin(), x_values.end());
    const double z_grid_min = *xminmax.first - 100.0;
    const double z_grid_max = *xminmax.second + 500.0;
    const double dz_grid = (z_grid_max - z_grid_min) / static_cast<double>(nz - 1);
    std::vector<double> amplitude(nz), density(nz), generated_on_active(steps);

    out.total.resize(x_values.size());
    out.electron.resize(x_values.size());
    out.hole.resize(x_values.size());
    out.no_offset.resize(x_values.size());
    out.offset = p[BM_scaleOffset] + p[SC_scaleOffset];

    for (std::size_t ix = 0; ix < x_values.size(); ++ix) {
        const double zc = x_values[ix];
        const double zR = zR_aberracion_esferica(
            zc, p[BM_zR0], p[BM_z_Aberr] + shift, p[BM_CoefA], p[BM_CoefB]
        );
        for (int iz = 0; iz < nz; ++iz) {
            const double z = z_grid_min + dz_grid * static_cast<double>(iz);
            const double scaled = (z - zc) / zR;
            const double intensity = 1.0 / (1.0 + scaled * scaled);
            amplitude[iz] = std::sqrt(std::max(intensity, 1e-15));
        }
        const double area_amplitude = trapezoid(amplitude, dz_grid);
        if (!std::isfinite(area_amplitude) || area_amplitude <= 0.0) {
            out.total[ix] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const double norm = p[BM_area] / area_amplitude;
        for (int iz = 0; iz < nz; ++iz) {
            const double amp = norm * amplitude[iz];
            density[iz] = amp * amp;
        }
        for (int ia = 0; ia < steps; ++ia) {
            generated_on_active[ia] = interp_uniform(density, z_grid_min, dz_grid, out.z[ia]) * p[SC_scaleAmp];
        }

        double electron_integral = 0.0;
        double hole_integral = 0.0;
        for (int ia = 0; ia + 1 < steps; ++ia) {
            electron_integral += 0.5 * (
                generated_on_active[ia] * out.response_e[ia]
                + generated_on_active[ia + 1] * out.response_e[ia + 1]
            ) * dz_active;
            hole_integral += 0.5 * (
                generated_on_active[ia] * out.response_h[ia]
                + generated_on_active[ia + 1] * out.response_h[ia + 1]
            ) * dz_active;
        }
        out.electron[ix] = p[BM_scaleAmp] * electron_integral;
        out.hole[ix] = p[BM_scaleAmp] * hole_integral;
        out.no_offset[ix] = out.electron[ix] + out.hole[ix];
        out.total[ix] = out.offset + out.no_offset[ix];
    }
    return out;
}

inline double threshold_duration(
    const std::vector<double>& waveform,
    double dt,
    double threshold_percent
) {
    if (waveform.size() < 3 || !std::isfinite(dt) || dt <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    auto peak_it = std::max_element(waveform.begin(), waveform.end());
    const std::size_t peak_index = static_cast<std::size_t>(std::distance(waveform.begin(), peak_it));
    const double peak = *peak_it;
    if (!std::isfinite(peak) || peak <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double level = peak * threshold_percent / 100.0;

    std::size_t left1 = 0;
    while (left1 < peak_index && waveform[left1] < level) ++left1;
    std::size_t left0 = left1 > 0 ? left1 - 1 : left1;
    if (left1 > peak_index) {
        left1 = peak_index;
        left0 = peak_index > 0 ? peak_index - 1 : peak_index;
    }

    std::size_t right1 = peak_index;
    while (right1 + 1 < waveform.size() && waveform[right1] > level) ++right1;
    std::size_t right0 = right1 > peak_index ? right1 - 1 : right1;

    auto crossing = [&](std::size_t i0, std::size_t i1) {
        const double y0 = waveform[i0];
        const double y1 = waveform[i1];
        const double t0 = static_cast<double>(i0) * dt;
        const double t1 = static_cast<double>(i1) * dt;
        if (!std::isfinite(y0) || !std::isfinite(y1) || y1 == y0) return t0;
        return t0 + (level - y0) * (t1 - t0) / (y1 - y0);
    };
    const double t_left = crossing(left0, left1);
    const double t_right = crossing(right0, right1);
    if (!std::isfinite(t_left) || !std::isfinite(t_right)) return std::numeric_limits<double>::quiet_NaN();
    return std::max(t_right - t_left, 0.0);
}

inline void rc_lowpass_inplace(std::vector<double>& waveform, double dt, double tau)
{
    if (waveform.empty() || !std::isfinite(dt) || dt <= 0.0 || !std::isfinite(tau) || tau <= 0.0) return;
    const double alpha = dt / (tau + dt);
    for (std::size_t i = 1; i < waveform.size(); ++i) {
        waveform[i] = waveform[i - 1] + alpha * (waveform[i] - waveform[i - 1]);
    }
}

std::vector<double> evaluate_tcoll_profile(
    const std::vector<double>& x_values,
    const std::vector<double>& p,
    int steps_per_active_region,
    int n_z_grid,
    bool voltage_enabled,
    int field_model_kind,
    double waveform_dt,
    double threshold_percent,
    double rc_tau,
    bool ignore_compute_wfs
) {
    std::vector<double> tcoll(x_values.size(), 1.0);
    if (ignore_compute_wfs) return tcoll;
    if (x_values.empty() || !std::isfinite(waveform_dt) || waveform_dt <= 0.0) return tcoll;

    const auto base = evaluate_profile(
        x_values, p, steps_per_active_region, n_z_grid, voltage_enabled, field_model_kind
    );
    if (base.z.size() < 2 || base.vdrift_mue.size() != base.z.size() || base.vdrift_muh.size() != base.z.size()) {
        tcoll.assign(x_values.size(), std::numeric_limits<double>::quiet_NaN());
        return tcoll;
    }

    const int steps = static_cast<int>(base.z.size());
    const int nz = std::max(n_z_grid, 2);
    const double width = p[BM_zRight];
    const double dz_active = width / static_cast<double>(steps - 1);

    double max_drift_time = 1.0;
    for (int start = 0; start < steps; ++start) {
        double te = 0.0;
        for (int seg = start; seg + 1 < steps; ++seg) {
            const double v = 0.5 * (base.vdrift_mue[seg] + base.vdrift_mue[seg + 1]);
            if (!std::isfinite(v) || v <= 0.0) break;
            te += dz_active / v;
        }
        double th = 0.0;
        for (int seg = start - 1; seg >= 0; --seg) {
            const double v = 0.5 * (base.vdrift_muh[seg] + base.vdrift_muh[seg + 1]);
            if (!std::isfinite(v) || v <= 0.0) break;
            th += dz_active / v;
        }
        if (std::isfinite(te)) max_drift_time = std::max(max_drift_time, te);
        if (std::isfinite(th)) max_drift_time = std::max(max_drift_time, th);
    }
    const double t_max = max_drift_time + 8.0 * std::max(rc_tau, 0.0) + 5.0 * waveform_dt;
    const int n_time = std::max(8, static_cast<int>(std::ceil(t_max / waveform_dt)) + 1);
    if (n_time > 200000) {
        tcoll.assign(x_values.size(), std::numeric_limits<double>::quiet_NaN());
        return tcoll;
    }

    const auto xminmax = std::minmax_element(x_values.begin(), x_values.end());
    const double z_grid_min = *xminmax.first - 100.0;
    const double z_grid_max = *xminmax.second + 500.0;
    const double dz_grid = (z_grid_max - z_grid_min) / static_cast<double>(nz - 1);
    std::vector<double> amplitude(nz), density(nz), generated_on_active(steps);
    std::vector<double> waveform(n_time);

    for (std::size_t ix = 0; ix < x_values.size(); ++ix) {
        const double zc = x_values[ix];
        const double zR = zR_aberracion_esferica(
            zc, p[BM_zR0], p[BM_z_Aberr] + p[SC_scale_zShift], p[BM_CoefA], p[BM_CoefB]
        );
        for (int iz = 0; iz < nz; ++iz) {
            const double z = z_grid_min + dz_grid * static_cast<double>(iz);
            const double scaled = (z - zc) / zR;
            const double intensity = 1.0 / (1.0 + scaled * scaled);
            amplitude[iz] = std::sqrt(std::max(intensity, 1e-15));
        }
        const double area_amplitude = trapezoid(amplitude, dz_grid);
        if (!std::isfinite(area_amplitude) || area_amplitude <= 0.0) {
            tcoll[ix] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const double norm = p[BM_area] / area_amplitude;
        for (int iz = 0; iz < nz; ++iz) {
            const double amp = norm * amplitude[iz];
            density[iz] = amp * amp;
        }
        for (int ia = 0; ia < steps; ++ia) {
            generated_on_active[ia] = interp_uniform(density, z_grid_min, dz_grid, base.z[ia]) * p[SC_scaleAmp];
        }

        std::fill(waveform.begin(), waveform.end(), 0.0);
        for (int start = 0; start < steps; ++start) {
            double weight = generated_on_active[start] * dz_active;
            if (start == 0 || start + 1 == steps) weight *= 0.5;
            if (!std::isfinite(weight) || weight <= 0.0) continue;

            double t_elapsed = 0.0;
            double survival = 1.0;
            for (int seg = start; seg + 1 < steps; ++seg) {
                const double v_mid = 0.5 * (base.vdrift_mue[seg] + base.vdrift_mue[seg + 1]);
                if (!std::isfinite(v_mid) || v_mid <= 0.0) break;
                const double dt_seg = dz_active / v_mid;
                const double t_mid = t_elapsed + 0.5 * dt_seg;
                const int index = static_cast<int>(std::floor(t_mid / waveform_dt));
                if (index >= 0 && index < n_time) waveform[index] += weight * survival * v_mid / width;
                survival *= (std::isfinite(p[TR_tau_e]) && p[TR_tau_e] > 0.0) ? std::exp(-dt_seg / p[TR_tau_e]) : 0.0;
                t_elapsed += dt_seg;
                if (survival <= 0.0) break;
            }

            t_elapsed = 0.0;
            survival = 1.0;
            for (int seg = start - 1; seg >= 0; --seg) {
                const double v_mid = 0.5 * (base.vdrift_muh[seg] + base.vdrift_muh[seg + 1]);
                if (!std::isfinite(v_mid) || v_mid <= 0.0) break;
                const double dt_seg = dz_active / v_mid;
                const double t_mid = t_elapsed + 0.5 * dt_seg;
                const int index = static_cast<int>(std::floor(t_mid / waveform_dt));
                if (index >= 0 && index < n_time) waveform[index] += weight * survival * v_mid / width;
                survival *= (std::isfinite(p[TR_tau_h]) && p[TR_tau_h] > 0.0) ? std::exp(-dt_seg / p[TR_tau_h]) : 0.0;
                t_elapsed += dt_seg;
                if (survival <= 0.0) break;
            }
        }
        rc_lowpass_inplace(waveform, waveform_dt, rc_tau);
        tcoll[ix] = threshold_duration(waveform, waveform_dt, threshold_percent);
    }
    return tcoll;
}

} // namespace trapping_roofit_v5

class TrappingGaussianNLLV5 : public RooAbsReal {
public:
    TrappingGaussianNLLV5() {}

    TrappingGaussianNLLV5(
        const char* name,
        const char* title,
        RooArgList& parameters,
        const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& sigma,
        const std::vector<int>& constraint_index,
        const std::vector<double>& constraint_mean,
        const std::vector<double>& constraint_sigma,
        int steps_per_active_region,
        int n_z_grid,
        bool voltage_enabled,
        int field_model_kind
    )
        : RooAbsReal(name, title),
          parameters_("parameters", "parameters", this),
          x_(x),
          y_(y),
          sigma_(sigma),
          constraint_index_(constraint_index),
          constraint_mean_(constraint_mean),
          constraint_sigma_(constraint_sigma),
          steps_per_active_region_(steps_per_active_region),
          n_z_grid_(n_z_grid),
          voltage_enabled_(voltage_enabled),
          field_model_kind_(field_model_kind)
    {
        parameters_.add(parameters);
    }

    TrappingGaussianNLLV5(const TrappingGaussianNLLV5& other, const char* name = nullptr)
        : RooAbsReal(other, name),
          parameters_("parameters", this, other.parameters_),
          x_(other.x_),
          y_(other.y_),
          sigma_(other.sigma_),
          constraint_index_(other.constraint_index_),
          constraint_mean_(other.constraint_mean_),
          constraint_sigma_(other.constraint_sigma_),
          steps_per_active_region_(other.steps_per_active_region_),
          n_z_grid_(other.n_z_grid_),
          voltage_enabled_(other.voltage_enabled_),
          field_model_kind_(other.field_model_kind_)
    {}

    TObject* clone(const char* newname) const override { return new TrappingGaussianNLLV5(*this, newname); }

protected:
    double evaluate() const override
    {
        std::vector<double> p(trapping_roofit_v5::N_PARAMETERS);
        for (int i = 0; i < trapping_roofit_v5::N_PARAMETERS; ++i) {
            const auto* value = dynamic_cast<const RooAbsReal*>(parameters_.at(i));
            if (!value) return 1e100;
            p[i] = value->getVal();
        }

        const auto profile = trapping_roofit_v5::evaluate_profile(
            x_, p, steps_per_active_region_, n_z_grid_, voltage_enabled_, field_model_kind_
        );
        if (profile.total.size() != y_.size()) return 1e100;

        double nll = 0.0;
        constexpr double log_sqrt_2pi = 0.91893853320467274178;
        for (std::size_t i = 0; i < y_.size(); ++i) {
            const double model = profile.total[i];
            const double sigma = sigma_[i];
            if (!std::isfinite(model) || !std::isfinite(sigma) || sigma <= 0.0) return 1e100;
            const double pull = (y_[i] - model) / sigma;
            nll += 0.5 * pull * pull + std::log(sigma) + log_sqrt_2pi;
        }
        for (std::size_t i = 0; i < constraint_index_.size(); ++i) {
            const int index = constraint_index_[i];
            if (index < 0 || index >= trapping_roofit_v5::N_PARAMETERS) continue;
            const double sigma = constraint_sigma_[i];
            if (!std::isfinite(sigma) || sigma <= 0.0) return 1e100;
            const double pull = (p[index] - constraint_mean_[i]) / sigma;
            nll += 0.5 * pull * pull + std::log(sigma) + log_sqrt_2pi;
        }
        return nll;
    }

private:
    RooListProxy parameters_;
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> sigma_;
    std::vector<int> constraint_index_;
    std::vector<double> constraint_mean_;
    std::vector<double> constraint_sigma_;
    int steps_per_active_region_ = 400;
    int n_z_grid_ = 20001;
    bool voltage_enabled_ = false;
    int field_model_kind_ = 0;
};

class TrappingQTCollGaussianNLLV1 : public RooAbsReal {
public:
    TrappingQTCollGaussianNLLV1() {}

    TrappingQTCollGaussianNLLV1(
        const char* name,
        const char* title,
        RooArgList& parameters,
        const std::vector<double>& x_q,
        const std::vector<double>& y_q,
        const std::vector<double>& sigma_q,
        const std::vector<double>& x_t,
        const std::vector<double>& y_t,
        const std::vector<double>& sigma_t,
        const std::vector<int>& constraint_index,
        const std::vector<double>& constraint_mean,
        const std::vector<double>& constraint_sigma,
        int steps_per_active_region,
        int n_z_grid,
        bool voltage_enabled,
        int field_model_kind,
        bool include_tcoll,
        bool ignore_compute_wfs,
        double waveform_dt,
        double threshold_percent,
        double rc_tau
    )
        : RooAbsReal(name, title),
          parameters_("parameters", "parameters", this),
          x_q_(x_q),
          y_q_(y_q),
          sigma_q_(sigma_q),
          x_t_(x_t),
          y_t_(y_t),
          sigma_t_(sigma_t),
          constraint_index_(constraint_index),
          constraint_mean_(constraint_mean),
          constraint_sigma_(constraint_sigma),
          steps_per_active_region_(steps_per_active_region),
          n_z_grid_(n_z_grid),
          voltage_enabled_(voltage_enabled),
          field_model_kind_(field_model_kind),
          include_tcoll_(include_tcoll),
          ignore_compute_wfs_(ignore_compute_wfs),
          waveform_dt_(waveform_dt),
          threshold_percent_(threshold_percent),
          rc_tau_(rc_tau)
    {
        parameters_.add(parameters);
    }

    TrappingQTCollGaussianNLLV1(const TrappingQTCollGaussianNLLV1& other, const char* name = nullptr)
        : RooAbsReal(other, name),
          parameters_("parameters", this, other.parameters_),
          x_q_(other.x_q_),
          y_q_(other.y_q_),
          sigma_q_(other.sigma_q_),
          x_t_(other.x_t_),
          y_t_(other.y_t_),
          sigma_t_(other.sigma_t_),
          constraint_index_(other.constraint_index_),
          constraint_mean_(other.constraint_mean_),
          constraint_sigma_(other.constraint_sigma_),
          steps_per_active_region_(other.steps_per_active_region_),
          n_z_grid_(other.n_z_grid_),
          voltage_enabled_(other.voltage_enabled_),
          field_model_kind_(other.field_model_kind_),
          include_tcoll_(other.include_tcoll_),
          ignore_compute_wfs_(other.ignore_compute_wfs_),
          waveform_dt_(other.waveform_dt_),
          threshold_percent_(other.threshold_percent_),
          rc_tau_(other.rc_tau_)
    {}

    TObject* clone(const char* newname) const override { return new TrappingQTCollGaussianNLLV1(*this, newname); }

protected:
    double evaluate() const override
    {
        std::vector<double> p(trapping_roofit_v5::N_PARAMETERS);
        for (int i = 0; i < trapping_roofit_v5::N_PARAMETERS; ++i) {
            const auto* value = dynamic_cast<const RooAbsReal*>(parameters_.at(i));
            if (!value) return 1e100;
            p[i] = value->getVal();
        }

        constexpr double log_sqrt_2pi = 0.91893853320467274178;
        double nll = 0.0;

        const auto profile = trapping_roofit_v5::evaluate_profile(
            x_q_, p, steps_per_active_region_, n_z_grid_, voltage_enabled_, field_model_kind_
        );
        if (profile.total.size() != y_q_.size()) return 1e100;
        for (std::size_t i = 0; i < y_q_.size(); ++i) {
            const double model = profile.total[i];
            const double sigma = sigma_q_[i];
            if (!std::isfinite(model) || !std::isfinite(sigma) || sigma <= 0.0) return 1e100;
            const double pull = (y_q_[i] - model) / sigma;
            nll += 0.5 * pull * pull + std::log(sigma) + log_sqrt_2pi;
        }

        if (include_tcoll_) {
            const auto tcoll = trapping_roofit_v5::evaluate_tcoll_profile(
                x_t_, p, steps_per_active_region_, n_z_grid_, voltage_enabled_, field_model_kind_,
                waveform_dt_, threshold_percent_, rc_tau_, ignore_compute_wfs_
            );
            if (tcoll.size() != y_t_.size()) return 1e100;
            for (std::size_t i = 0; i < y_t_.size(); ++i) {
                const double model = tcoll[i];
                const double sigma = sigma_t_[i];
                if (!std::isfinite(model) || !std::isfinite(sigma) || sigma <= 0.0) return 1e100;
                const double pull = (y_t_[i] - model) / sigma;
                nll += 0.5 * pull * pull + std::log(sigma) + log_sqrt_2pi;
            }
        }

        for (std::size_t i = 0; i < constraint_index_.size(); ++i) {
            const int index = constraint_index_[i];
            if (index < 0 || index >= trapping_roofit_v5::N_PARAMETERS) continue;
            const double sigma = constraint_sigma_[i];
            if (!std::isfinite(sigma) || sigma <= 0.0) return 1e100;
            const double pull = (p[index] - constraint_mean_[i]) / sigma;
            nll += 0.5 * pull * pull + std::log(sigma) + log_sqrt_2pi;
        }
        return nll;
    }

private:
    RooListProxy parameters_;
    std::vector<double> x_q_;
    std::vector<double> y_q_;
    std::vector<double> sigma_q_;
    std::vector<double> x_t_;
    std::vector<double> y_t_;
    std::vector<double> sigma_t_;
    std::vector<int> constraint_index_;
    std::vector<double> constraint_mean_;
    std::vector<double> constraint_sigma_;
    int steps_per_active_region_ = 400;
    int n_z_grid_ = 20001;
    bool voltage_enabled_ = false;
    int field_model_kind_ = 0;
    bool include_tcoll_ = false;
    bool ignore_compute_wfs_ = false;
    double waveform_dt_ = 0.02;
    double threshold_percent_ = 5.0;
    double rc_tau_ = 1.6;
};
        '''
    )
    return ROOT


def _parameter_values(parameter_specs, fit_names=None, fit_vector=None):
    return _base._parameter_values(parameter_specs, fit_names, fit_vector)


def _effective_fit_names(parameter_specs):
    disabled = {"EF_CoefC"}
    if bool(parameter_specs["EF_BiasVoltage"].get("enabled", False)):
        disabled.add("EF_ExpAmpRight")
    return [name for name in _base._effective_fit_names(parameter_specs) if name not in disabled]


def _resolved_fit_bounds(parameter_specs, fit_names, fit_options):
    return _base._resolved_fit_bounds(parameter_specs, fit_names, fit_options)


def _configuration_values(config):
    values = _parameter_values(config["parameters"])
    _apply_derived_field_values(values, config)
    return np.array([values[name] for name in MODEL_PARAMETER_NAMES], dtype=float)


def _apply_derived_field_values(values, config):
    model = _canonical_field_model(config["fit_options"].get("field_model", "polynomial"))
    if bool(config["parameters"]["EF_BiasVoltage"].get("enabled", False)):
        if model == "double_exponential":
            values["EF_ExpAmpRight"] = _derived_exp_amp_right(values)
        else:
            values["EF_CoefC"] = _derived_field_constant(values)
    return values


def _derived_field_constant(values):
    width = float(values["BM_zRight"])
    if not np.isfinite(width) or width == 0.0:
        raise ValueError("BM_zRight must be finite and non-zero")
    center = float(values["EF_z0"])
    u_left = -center
    u_right = width - center
    quadratic_integral = float(values["EF_CoefA"]) * (u_right**3 - u_left**3) / 3.0
    linear_integral = float(values["EF_CoefB"]) * (u_right**2 - u_left**2) / 2.0
    return (float(values["EF_BiasVoltage"]) - quadratic_integral - linear_integral) / width


def _field_voltage_from_coefficients(values):
    width = float(values["BM_zRight"])
    center = float(values["EF_z0"])
    u_left = -center
    u_right = width - center
    return (
        float(values["EF_CoefA"]) * (u_right**3 - u_left**3) / 3.0
        + float(values["EF_CoefB"]) * (u_right**2 - u_left**2) / 2.0
        + float(values["EF_CoefC"]) * width
    )


def _exp_integral(width, decay):
    decay = float(decay)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("BM_zRight must be finite and positive")
    if not np.isfinite(decay) or decay <= 0.0:
        raise ValueError("Double-exponential decay lengths must be finite and positive")
    return decay * (1.0 - np.exp(-width / decay))


def _derived_exp_amp_right(values):
    width = float(values["BM_zRight"])
    left_integral = _exp_integral(width, values["EF_ExpDecayLeft"])
    right_integral = _exp_integral(width, values["EF_ExpDecayRight"])
    if right_integral <= 0.0:
        raise ValueError("EF_ExpDecayRight gives a non-positive field integral")
    return (
        float(values["EF_BiasVoltage"])
        - float(values["EF_ExpAmpLeft"]) * left_integral
    ) / right_integral


def _field_voltage_from_double_exponential(values):
    width = float(values["BM_zRight"])
    return (
        float(values["EF_ExpAmpLeft"]) * _exp_integral(width, values["EF_ExpDecayLeft"])
        + float(values["EF_ExpAmpRight"]) * _exp_integral(width, values["EF_ExpDecayRight"])
    )


def _to_numpy(vector):
    return np.array([float(vector[i]) for i in range(vector.size())], dtype=float)


def _profile_result_to_dict(result):
    response = {
        "z": _to_numpy(result.z),
        "efield": _to_numpy(result.efield),
        "mobility_e": _to_numpy(result.mobility_e),
        "mobility_h": _to_numpy(result.mobility_h),
        "vdrift_mue": _to_numpy(result.vdrift_mue),
        "vdrift_muh": _to_numpy(result.vdrift_muh),
        "response_e": _to_numpy(result.response_e),
        "response_h": _to_numpy(result.response_h),
        "response_total": _to_numpy(result.response_total),
    }
    return (
        _to_numpy(result.total),
        _to_numpy(result.electron),
        _to_numpy(result.hole),
        _to_numpy(result.no_offset),
        float(result.offset),
        response,
    )


def simulate_trapping_model(x_vec, configuration):
    config = load_fit_configuration(configuration)
    ROOT = _compile_cpp_model()
    options = config["fit_options"]
    result = ROOT.trapping_roofit_v5.evaluate_profile(
        _std_vector(x_vec),
        _std_vector(_configuration_values(config)),
        int(options.get("steps_per_active_region", 400)),
        int(options.get("n_z_grid", DEFAULT_N_Z_GRID)),
        bool(config["parameters"]["EF_BiasVoltage"].get("enabled", False)),
        _field_model_kind(config),
    )
    return _profile_result_to_dict(result)


def _make_point_sigmas(y_data, supplied_sigma, options):
    data = np.asarray(y_data, dtype=float)
    if supplied_sigma is not None:
        sigma = np.asarray(supplied_sigma, dtype=float)
    else:
        fraction = float(options.get("relative_sigma_fraction", 0.02))
        floor_fraction = float(options.get("relative_sigma_floor_fraction", 1e-6))
        scale = max(float(np.nanmax(np.abs(data))), 1.0)
        sigma = fraction * np.maximum(np.abs(data), floor_fraction * scale)
    if sigma.shape != data.shape or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Point sigmas must match y_data and contain finite positive values")
    return sigma


def _make_roofit_parameters(ROOT, config):
    specs = config["parameters"]
    options = config["fit_options"]
    fit_names = _effective_fit_names(specs)
    fit_set = set(fit_names)
    lower, upper = _resolved_fit_bounds(specs, fit_names, options) if fit_names else (np.array([]), np.array([]))
    bound_by_name = {
        name: (float(lower[index]), float(upper[index]))
        for index, name in enumerate(fit_names)
    }

    variables = {}
    parameter_list = ROOT.RooArgList()
    constraint_indices = []
    constraint_means = []
    constraint_sigmas = []
    range_sigmas = float(options.get("constraint_range_sigmas", 10.0))

    for index, name in enumerate(MODEL_PARAMETER_NAMES):
        spec = specs[name]
        value = float(spec["value"])
        if name in bound_by_name:
            lo, hi = bound_by_name[name]
        else:
            lo, hi = -np.inf, np.inf

        constraint = spec.get("constraint")
        if name in fit_set and constraint is not None:
            mean = float(constraint["mean"])
            sigma = float(constraint["sigma"])
            if not np.isfinite(lo):
                lo = mean - range_sigmas * sigma
            if not np.isfinite(hi):
                hi = mean + range_sigmas * sigma

        if np.isfinite(lo) and np.isfinite(hi):
            var = ROOT.RooRealVar(name, parameter_label(name), value, lo, hi)
        else:
            var = ROOT.RooRealVar(name, parameter_label(name), value)
            if np.isfinite(lo):
                var.setMin(lo)
            if np.isfinite(hi):
                var.setMax(hi)
        var.setConstant(name not in fit_set)
        variables[name] = var
        parameter_list.add(var)

        if constraint is not None:
            constraint_indices.append(index)
            constraint_means.append(float(constraint["mean"]))
            constraint_sigmas.append(float(constraint["sigma"]))

    return variables, parameter_list, fit_names, bound_by_name, constraint_indices, constraint_means, constraint_sigmas


def _current_parameter_values(variables, config=None):
    values = {}
    for name in MODEL_PARAMETER_NAMES:
        values[name] = float(variables[name].getVal())
    if config is None:
        values["EF_CoefC"] = _derived_field_constant(values)
        values["EF_BiasVoltage"] = _field_voltage_from_coefficients(values)
    else:
        _apply_derived_field_values(values, config)
        if bool(config["parameters"]["EF_BiasVoltage"].get("enabled", False)):
            model = _canonical_field_model(config["fit_options"].get("field_model", "polynomial"))
            if model == "double_exponential":
                values["EF_BiasVoltage"] = _field_voltage_from_double_exponential(values)
            else:
                values["EF_BiasVoltage"] = _field_voltage_from_coefficients(values)
    return values


def _covariance_and_correlation(ROOT, fit_result, fit_names):
    n = len(fit_names)
    covariance = np.full((n, n), np.nan, dtype=float)
    correlation = np.full((n, n), np.nan, dtype=float)
    if fit_result is None:
        return covariance, correlation
    try:
        cov = fit_result.covarianceMatrix()
        cor = fit_result.correlationMatrix()
        for i in range(n):
            for j in range(n):
                covariance[i, j] = float(cov[i][j])
                correlation[i, j] = float(cor[i][j])
    except Exception:
        pass
    return covariance, correlation


def _errors_from_roofit_vars(variables, fit_names):
    return np.array([float(variables[name].getError()) for name in fit_names], dtype=float)


def _minos_errors_from_roofit_vars(variables, fit_names):
    errors_minus = np.full(len(fit_names), np.nan, dtype=float)
    errors_plus = np.full(len(fit_names), np.nan, dtype=float)
    for index, name in enumerate(fit_names):
        error_low = float(variables[name].getErrorLo())
        error_high = float(variables[name].getErrorHi())
        if np.isfinite(error_low) and error_low < 0.0:
            errors_minus[index] = abs(error_low)
        if np.isfinite(error_high) and error_high > 0.0:
            errors_plus[index] = error_high
    return errors_minus, errors_plus


def _selected_minos_parameters(requested_names, fit_names):
    if requested_names is None:
        requested_names = []
    if isinstance(requested_names, str):
        requested_names = [requested_names]
    requested_names = [str(name) for name in requested_names]
    if not requested_names:
        return list(fit_names)
    unknown = [name for name in requested_names if name not in fit_names]
    if unknown:
        raise ValueError(
            "params_MINOS contains parameters that are not floating fit parameters: "
            + ", ".join(unknown)
        )
    return requested_names


def _decode_minos_status(status):
    if status is None:
        return []
    status = int(status)
    if status == 0:
        return ["ok"]
    if status < 0:
        return ["exception/no minimum"]
    meanings = []
    if status & 1:
        meanings.append("invalid lower")
    if status & 2:
        meanings.append("invalid upper")
    if status & 4:
        meanings.append("max calls")
    if status & 8:
        meanings.append("new minimum")
    if status & 16:
        meanings.append("at limit")
    known_bits = 1 | 2 | 4 | 8 | 16
    extra = status & ~known_bits
    if extra:
        meanings.append(f"unknown bits {extra}")
    return meanings


def _included_real_parameter_values(include_real):
    if include_real is None:
        return None
    if isinstance(include_real, (str, Path)):
        return _parameter_values(load_fit_configuration(include_real)["parameters"])
    if isinstance(include_real, dict) and "parameters" in include_real:
        return _parameter_values(load_fit_configuration(include_real)["parameters"])
    if isinstance(include_real, dict):
        values = {}
        for name in MODEL_PARAMETER_NAMES:
            value = include_real.get(name, np.nan)
            if isinstance(value, dict):
                value = value.get("value", np.nan)
            values[name] = float(value) if value is not None else np.nan
        return values
    raise TypeError("include_real must be None, a configuration path, a configuration dict, or a parameter-value dict")


def _format_sigfig(value, sigfigs):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    return f"{value:.{int(sigfigs)}g}"


def _summary_separator(width=160):
    return "_" * width


def _integrated_drift_time_ns(z_values, velocity_values):
    z_values = np.asarray(z_values, dtype=float)
    velocity_values = np.asarray(velocity_values, dtype=float)
    if z_values.shape != velocity_values.shape or z_values.size < 2:
        return np.nan
    finite = np.isfinite(z_values) & np.isfinite(velocity_values)
    if not np.all(finite) or np.any(velocity_values <= 0.0):
        return np.inf
    return float(np.trapz(1.0 / velocity_values, z_values))


def _format_drift_time(time_ns):
    if not np.isfinite(time_ns):
        return "inf"
    if abs(time_ns) >= 1e3:
        return f"{time_ns / 1e3:.3g} us"
    return f"{time_ns:.3g} ns"


def _print_fit_summary(output):
    sigfigs = int(output["configuration"]["fit_options"].get("table_sigfigs", 4))
    line = _summary_separator()
    print(f"NLL = {_format_sigfig(output['nll'], sigfigs)}")
    print(
        f"chi2 / dof = {_format_sigfig(output['chi2'], sigfigs)} / "
        f"{output['dof']} = {_format_sigfig(output['chi2_dof'], sigfigs)}"
    )
    timings = output.get("timings", {})
    if "minimize_elapsed_s" in timings:
        print(f"elapsed fit minimize = {_format_sigfig(timings['minimize_elapsed_s'], sigfigs)} s")
    if timings.get("minos_elapsed_s") is not None:
        print(f"elapsed MINOS = {_format_sigfig(timings['minos_elapsed_s'], sigfigs)} s")
    if bool(output["configuration"]["fit_options"].get("do_useMINOS", False)):
        minos_names = output.get("minos_names", [])
        minos_label = ", ".join(minos_names) if minos_names else "--"
        print(f"MINOS aggregate status = {output.get('minos_status')} ; parameters = {minos_label}")
        status_by_parameter = output.get("minos_status_by_parameter", {})
        if status_by_parameter:
            status_text = ", ".join(
                f"{name}:{status} ({'; '.join(_decode_minos_status(status))})"
                for name, status in status_by_parameter.items()
            )
            print(f"MINOS status by parameter = {status_text}")
    real_parameters = output.get("real_parameters")
    print(line)
    if real_parameters is None:
        header = (
            f"{'type':8s} {'parameter / unit':34s} {'initial':>13s} {'final':>13s} "
            f"{'error HESSE':>18s} {'error MINOS':>25s} {'constraint':>22s}"
        )
    else:
        header = (
            f"{'type':8s} {'parameter / unit':34s} {'initial':>13s} {'real':>13s} {'final':>13s} "
            f"{'error HESSE':>18s} {'error MINOS':>25s} {'constraint':>22s}"
        )
    print(header)
    print(line)
    fit_index = {name: index for index, name in enumerate(output["fit_names"])}
    bounds = output.get("fit_bounds", {})
    fit_names = [
        name for name in output["fit_names"]
        if output["configuration"]["parameters"][name].get("constraint") is None
    ]
    cstr_names = [
        name for name in output["fit_names"]
        if output["configuration"]["parameters"][name].get("constraint") is not None
    ]
    fixed_names = [name for name in MODEL_PARAMETER_NAMES if name not in fit_index]
    ordered_names = fit_names + cstr_names + fixed_names
    cstr_separator_printed = False
    fixed_separator_printed = False
    for name in ordered_names:
        if name in cstr_names and not cstr_separator_printed and fit_names:
            print(line)
            cstr_separator_printed = True
        if name in fixed_names and not fixed_separator_printed and (fit_names or cstr_names):
            print(line)
            fixed_separator_printed = True
        spec = output["configuration"]["parameters"][name]
        if name in fit_index:
            ptype = "cstr" if spec.get("constraint") is not None else "fit"
        else:
            ptype = "fixed"
        if name in fit_index:
            index = fit_index[name]
            minus = output["errors_minus"][index]
            plus = output["errors_plus"][index]
            hesse = output["errors"][index]
            hesse_text = _format_sigfig(hesse, sigfigs)
            minos_text = (
                f"-{_format_sigfig(minus, sigfigs)}/+{_format_sigfig(plus, sigfigs)}"
                if np.isfinite(minus) and np.isfinite(plus) else "--"
            )
            bound = bounds.get(name)
            if bound is not None:
                low, high = bound
                if low is not None and high is not None and np.isfinite(low) and np.isfinite(high):
                    value = float(output["parameters"][name])
                    scale = max(abs(high - low), 1.0)
                    if abs(value - low) <= 1e-4 * scale:
                        minos_text = f"{minos_text} [at min]"
                    elif abs(value - high) <= 1e-4 * scale:
                        minos_text = f"{minos_text} [at max]"
        else:
            hesse_text = "--"
            minos_text = "--"
        constraint = spec.get("constraint")
        constraint_text = "--" if constraint is None else (
            f"{_format_sigfig(constraint['mean'], sigfigs)} +/- {_format_sigfig(constraint['sigma'], sigfigs)}"
        )
        if real_parameters is None:
            print(
                f"{ptype:8s} {parameter_label(name):34s} "
                f"{_format_sigfig(spec['value'], sigfigs):>13s} {_format_sigfig(output['parameters'][name], sigfigs):>13s} "
                f"{hesse_text:>18s} {minos_text:>25s} {constraint_text:>22s}"
            )
        else:
            real_value = float(real_parameters.get(name, np.nan))
            print(
                f"{ptype:8s} {parameter_label(name):34s} "
                f"{_format_sigfig(spec['value'], sigfigs):>13s} {_format_sigfig(real_value, sigfigs):>13s} "
                f"{_format_sigfig(output['parameters'][name], sigfigs):>13s} "
                f"{hesse_text:>18s} {minos_text:>25s} {constraint_text:>22s}"
            )
    print(line)


def plot_fit_diagnostics(output):
    import matplotlib.pyplot as plt

    x = np.asarray(output["x"], dtype=float)
    response = output["material_response"]
    figure = plt.figure(figsize=(10.0, 5.5), constrained_layout=True)
    grid = figure.add_gridspec(3, 3)
    axis_profile = figure.add_subplot(grid[:2, :2])
    axis_residual = figure.add_subplot(grid[2, :2], sharex=axis_profile)
    axis_field = figure.add_subplot(grid[0, 2])
    axis_velocity = figure.add_subplot(grid[1, 2])
    axis_response = figure.add_subplot(grid[2, 2])

    profile_offset = float(output.get("profile_offset", output.get("y_fit_offset", 0.0)))
    y_data_plot = np.asarray(output["y_data"], dtype=float) - profile_offset
    y_fit_data_plot = np.asarray(output["y_fit"], dtype=float) - profile_offset
    y_fit_plot = y_fit_data_plot
    y_fit_e_plot = np.asarray(output["y_fit_e"], dtype=float)
    y_fit_h_plot = np.asarray(output["y_fit_h"], dtype=float)
    x_curve = x

    if x.size >= 2 and "configuration" in output and "parameters" in output:
        dense_size = max(int(x.size) * 10, int(x.size))
        x_curve = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), dense_size)
        curve_config = copy.deepcopy(output["configuration"])
        for name in MODEL_PARAMETER_NAMES:
            if name in curve_config["parameters"] and name in output["parameters"]:
                curve_config["parameters"][name]["value"] = float(output["parameters"][name])
        try:
            curve_total, curve_e, curve_h, _, curve_offset, _ = simulate_trapping_model(x_curve, curve_config)
            y_fit_plot = np.asarray(curve_total, dtype=float) - float(curve_offset)
            y_fit_e_plot = np.asarray(curve_e, dtype=float)
            y_fit_h_plot = np.asarray(curve_h, dtype=float)
        except Exception:
            x_curve = x

    axis_profile.errorbar(
        x, y_data_plot, yerr=output["y_sigma"], fmt="o", ms=3,
        color="black", ecolor="0.6", elinewidth=0.8, capsize=0, label="data",
    )
    axis_profile.plot(x_curve, y_fit_plot, color="tab:orange", lw=2,
                      label=f"fit, chi2/dof = {output['chi2_dof']:.3f}")
    if y_fit_e_plot.shape == x_curve.shape:
        axis_profile.plot(x_curve, y_fit_e_plot, color="darkblue", lw=1.5, alpha=0.8, label="e")
    if y_fit_h_plot.shape == x_curve.shape:
        axis_profile.plot(x_curve, y_fit_h_plot, color="crimson", lw=1.5, alpha=0.8, label="h")
    axis_profile.set(xlabel="focus position z / um", ylabel="charge / NE", title=f"RooFit ID: {output['generation_id']}")
    axis_profile.legend(frameon=False)

    residual = y_data_plot - y_fit_data_plot
    axis_residual.errorbar(x, residual, yerr=output["y_sigma"], fmt="o", ms=3, color="black", ecolor="0.6")
    axis_residual.axhline(0.0, color="black", lw=1)
    axis_residual.set(xlabel="focus position z / um", ylabel="data - fit / NE", title="residuals")

    response_z = np.asarray(response.get("z", []), dtype=float)
    response_v_e = np.asarray(response.get("vdrift_mue", []), dtype=float)
    response_v_h = np.asarray(response.get("vdrift_muh", []), dtype=float)
    response_total = np.asarray(response.get("response_total", []), dtype=float)
    response_e = np.asarray(response.get("response_e", []), dtype=float)
    response_h = np.asarray(response.get("response_h", []), dtype=float)
    response_field = np.asarray(response.get("efield", []), dtype=float)

    velocity_positive_values = []
    if response_v_e.shape == response_z.shape:
        time_e = _integrated_drift_time_ns(response_z, response_v_e)
        axis_velocity.plot(
            response_z, response_v_e, color="darkblue",
            label=f"e, t={_format_drift_time(time_e)}",
        )
        velocity_positive_values.extend(response_v_e[np.isfinite(response_v_e) & (response_v_e > 0.0)])
    if response_v_h.shape == response_z.shape:
        time_h = _integrated_drift_time_ns(response_z, response_v_h)
        axis_velocity.plot(
            response_z, response_v_h, color="crimson",
            label=f"h, t={_format_drift_time(time_h)}",
        )
        velocity_positive_values.extend(response_v_h[np.isfinite(response_v_h) & (response_v_h > 0.0)])
    axis_velocity.set(title="drift velocity", xlabel="z / um", ylabel="velocity / (um/ns)")
    if velocity_positive_values:
        velocity_positive_values = np.asarray(velocity_positive_values, dtype=float)
        axis_velocity.set_yscale("log")
        axis_velocity.set_ylim(
            max(float(np.nanmin(velocity_positive_values)) * 0.5, 1e-6),
            float(np.nanmax(velocity_positive_values)) * 2.0,
        )
    if axis_velocity.get_legend_handles_labels()[0]:
        axis_velocity.legend(frameon=False)
    if response_total.shape == response_z.shape:
        axis_response.plot(response_z, response_total, color="black", label="sum")
    if response_e.shape == response_z.shape:
        axis_response.plot(response_z, response_e, color="darkblue", alpha=0.8, label="e")
    if response_h.shape == response_z.shape:
        axis_response.plot(response_z, response_h, color="crimson", alpha=0.8, label="h")
    axis_response.set(title="material response", xlabel="z / um", ylabel="response / adim.")
    if axis_response.get_legend_handles_labels()[0]:
        axis_response.legend(frameon=False)
    if response_field.shape == response_z.shape:
        axis_field.plot(response_z, response_field, color="tab:green")
    axis_field.set(title="electric field", xlabel="z / um", ylabel="field / (V/um)")
    for axis in figure.axes:
        axis.grid(alpha=0.25)
    return figure


def plot_fit_correlation(output):
    import matplotlib.pyplot as plt

    correlation = output["correlation"]
    labels = [parameter_label(name) for name in output["fit_names"]]
    figure, axis = plt.subplots(figsize=(max(6, 0.7 * len(labels)), max(5, 0.65 * len(labels))))
    image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("RooFit correlation matrix")
    figure.colorbar(image, ax=axis, label="correlation")
    figure.tight_layout()
    return figure


def _save_fit_result(output, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{output['generation_id']}.json"
    fit_names = list(output["fit_names"])
    hesse_errors = {
        name: float(output["errors"][index])
        for index, name in enumerate(fit_names)
    }
    minos_errors = {
        name: {
            "minus": (
                float(output["errors_minus"][index])
                if np.isfinite(output["errors_minus"][index]) else None
            ),
            "plus": (
                float(output["errors_plus"][index])
                if np.isfinite(output["errors_plus"][index]) else None
            ),
            "status": output["minos_status_by_parameter"].get(name),
            "raw_status": output["minos_raw_status_by_parameter"].get(name),
        }
        for index, name in enumerate(fit_names)
    }
    payload = {
        "format_version": 1,
        "generation_id": output["generation_id"],
        "fit_name": output["fit_name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "model_library": "trapping_model_Q_TColl_li.py",
            "python": sys.version.split()[0],
        },
        "fit_summary": {
            "minimize_status": output["minimize_status"],
            "minos_status": output["minos_status"],
            "minos_names": output["minos_names"],
            "minos_status_by_parameter": output["minos_status_by_parameter"],
            "minos_raw_status_by_parameter": output["minos_raw_status_by_parameter"],
            "nll": output["nll"],
            "chi2": output["chi2"],
            "dof": output["dof"],
            "chi2_dof": output["chi2_dof"],
            "timings": output["timings"],
            "refit_fromLocalMinimumMINOS": output["refit_fromLocalMinimumMINOS"],
        },
        "configuration": output["configuration"],
        "initial_parameters": output["initial_parameters"],
        "real_parameters": output["real_parameters"],
        "final_parameters": output["parameters"],
        "fit_parameters": {
            "names": fit_names,
            "bounds": output["fit_bounds"],
            "hesse_errors": hesse_errors,
            "minos_errors": minos_errors,
            "covariance": np.asarray(output["covariance"], dtype=float).tolist(),
            "correlation": np.asarray(output["correlation"], dtype=float).tolist(),
        },
        "data": {
            "x_data": np.asarray(output["x"], dtype=float).tolist(),
            "y_data": np.asarray(output["y_data"], dtype=float).tolist(),
            "y_sigma": np.asarray(output["y_sigma"], dtype=float).tolist(),
            "y_sigma_supplied": bool(output["y_sigma_supplied"]),
        },
        "model_curves": {
            "y_fit": np.asarray(output["y_fit"], dtype=float).tolist(),
            "y_fit_e": np.asarray(output["y_fit_e"], dtype=float).tolist(),
            "y_fit_h": np.asarray(output["y_fit_h"], dtype=float).tolist(),
            "y_fit_no_offset": np.asarray(output["y_fit_no_offset"], dtype=float).tolist(),
            "y_fit_offset": float(output["y_fit_offset"]),
            "profile_offset": float(output["profile_offset"]),
        },
        "material_response": output["material_response"],
        "minos_new_minimum_candidates": output["minos_new_minimum_candidates"],
        "artifacts": {
            "diagnostic_plot_png": (
                str(output["diagnostic_plot_path"])
                if output.get("diagnostic_plot_path") is not None else None
            ),
            "json": str(json_path),
        },
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    output["json_path"] = json_path
    return json_path


def _save_fit_diagnostic_plot(output, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / f"{output['generation_id']}_diagnostic.png"
    figure = plot_fit_diagnostics(output)
    figure.savefig(png_path, dpi=180)
    try:
        import matplotlib.pyplot as plt
        plt.close(figure)
    except Exception:
        pass
    output["diagnostic_plot_path"] = png_path
    return png_path


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
    save_results=False,
    fit_name="roofit",
    output_dir="SiC_UVLED/Fits_RooFit",
    do_useMINOS=None,
    params_MINOS=None,
    include_real=None,
    table_sigfigs=None,
    refit_fromLocalMinimumMINOS=None,
    show_roofit_output=None,
    _minos_refit_depth=0,
):
    total_start = perf_counter()
    ROOT = _compile_cpp_model()
    config = load_fit_configuration(configuration)
    specs = config["parameters"]
    options = config["fit_options"]
    if do_useMINOS is not None:
        options["do_useMINOS"] = bool(do_useMINOS)
    if params_MINOS is not None:
        options["params_MINOS"] = params_MINOS
    if table_sigfigs is not None:
        table_sigfigs = int(table_sigfigs)
        if table_sigfigs <= 0:
            raise ValueError("table_sigfigs must be a positive integer")
        options["table_sigfigs"] = table_sigfigs
    if refit_fromLocalMinimumMINOS is not None:
        options["refit_fromLocalMinimumMINOS"] = bool(refit_fromLocalMinimumMINOS)
    if show_roofit_output is not None:
        options["show_roofit_output"] = bool(show_roofit_output)
    real_parameters = _included_real_parameter_values(include_real)

    x = np.asarray(x_vec, dtype=float)
    data = np.asarray(y_data, dtype=float)
    if x.ndim != 1 or data.shape != x.shape:
        raise ValueError("x_vec and y_data must be one-dimensional arrays with equal shape")
    sigma_supplied = y_sigma is not None
    sigma = _make_point_sigmas(data, y_sigma, options)

    variables, parameter_list, fit_names, bounds, constraint_indices, constraint_means, constraint_sigmas = _make_roofit_parameters(ROOT, config)
    if not fit_names:
        raise ValueError("At least one effective parameter must have type='fit'")

    nll = ROOT.TrappingGaussianNLLV5(
        "trapping_gaussian_nll",
        "trapping Gaussian NLL",
        parameter_list,
        _std_vector(x),
        _std_vector(data),
        _std_vector(sigma),
        ROOT.std.vector("int")(constraint_indices),
        _std_vector(constraint_means),
        _std_vector(constraint_sigmas),
        int(options.get("steps_per_active_region", 400)),
        int(options.get("n_z_grid", DEFAULT_N_Z_GRID)),
        bool(specs["EF_BiasVoltage"].get("enabled", False)),
        _field_model_kind(config),
    )

    minimizer = ROOT.RooMinimizer(nll)
    minimizer.setMinimizerType(str(options.get("minimizer_type", "Minuit2")))
    minimizer.setStrategy(int(options.get("strategy", 1)))
    minimizer.setPrintLevel(int(options.get("print_level", -1)))
    minimize_start = perf_counter()
    with _suppress_c_output(not bool(options.get("show_roofit_output", False))):
        status = int(minimizer.minimize(
            str(options.get("minimizer_type", "Minuit2")),
            str(options.get("minimizer_algorithm", "Minimize")),
        ))
    minimize_elapsed = perf_counter() - minimize_start
    hesse_errors = _errors_from_roofit_vars(variables, fit_names)
    best_fit_values = {name: float(variables[name].getVal()) for name in MODEL_PARAMETER_NAMES}
    best_fit_nll = float(nll.getVal())
    minos_status = None
    minos_names = []
    minos_elapsed = None
    minos_status_by_parameter = {}
    minos_raw_status_by_parameter = {}
    minos_elapsed_by_parameter = {}
    minos_new_minimum_candidates = {}
    refit_candidate_name = None
    refit_candidate_values = None
    errors_minus = np.full(len(fit_names), np.nan, dtype=float)
    errors_plus = np.full(len(fit_names), np.nan, dtype=float)
    if bool(options.get("do_useMINOS", False)):
        minos_names = _selected_minos_parameters(options.get("params_MINOS", []), fit_names)
        minos_start = perf_counter()
        fit_index = {name: index for index, name in enumerate(fit_names)}
        previous_raw_minos_status = int(status)
        for name in minos_names:
            for parameter_name, value in best_fit_values.items():
                variables[parameter_name].setVal(value)
            minos_parameters = ROOT.RooArgSet()
            minos_parameters.add(variables[name])
            parameter_start = perf_counter()
            with _suppress_c_output(not bool(options.get("show_roofit_output", False))):
                try:
                    raw_parameter_status = int(minimizer.minos(minos_parameters))
                except Exception:
                    raw_parameter_status = -1
            parameter_elapsed = perf_counter() - parameter_start
            if raw_parameter_status < 0:
                parameter_status = raw_parameter_status
            else:
                raw_delta = raw_parameter_status - previous_raw_minos_status
                if raw_delta >= 0 and raw_delta % 10 == 0:
                    parameter_status = raw_delta // 10
                else:
                    parameter_status = raw_parameter_status
            previous_raw_minos_status = raw_parameter_status if raw_parameter_status >= 0 else previous_raw_minos_status
            minos_raw_status_by_parameter[name] = raw_parameter_status
            minos_status_by_parameter[name] = parameter_status
            minos_elapsed_by_parameter[name] = parameter_elapsed
            if parameter_status > 0 and (parameter_status & 8):
                candidate_values = _current_parameter_values(variables, config)
                candidate_nll = float(nll.getVal())
                minos_new_minimum_candidates[name] = {
                    "nll": candidate_nll,
                    "delta_nll": candidate_nll - best_fit_nll,
                    "parameters": candidate_values,
                }
                if refit_candidate_name is None:
                    refit_candidate_name = name
                    refit_candidate_values = candidate_values
            if parameter_status == 0:
                error_low = float(variables[name].getErrorLo())
                error_high = float(variables[name].getErrorHi())
                index = fit_index[name]
                if np.isfinite(error_low) and error_low < 0.0:
                    errors_minus[index] = abs(error_low)
                if np.isfinite(error_high) and error_high > 0.0:
                    errors_plus[index] = error_high
        minos_elapsed = perf_counter() - minos_start
        minos_status = 0 if all(status == 0 for status in minos_status_by_parameter.values()) else 1
        for parameter_name, value in best_fit_values.items():
            variables[parameter_name].setVal(value)
    if (
        bool(options.get("refit_fromLocalMinimumMINOS", False))
        and refit_candidate_values is not None
        and _minos_refit_depth == 0
    ):
        print(
            "WARNING: MINOS found a new local minimum while profiling "
            f"{refit_candidate_name}; refitting from that parameter combination."
        )
        refit_config = copy.deepcopy(config)
        for parameter_name in fit_names:
            value = float(refit_candidate_values[parameter_name])
            if np.isfinite(value):
                refit_config["parameters"][parameter_name]["value"] = value
        refit_output = fit_trapping_model(
            x,
            data,
            refit_config,
            y_sigma=sigma,
            show_summary=show_summary,
            show_fit=show_fit,
            show_covariance=show_covariance,
            show_correlation=show_correlation,
            save_results=save_results,
            fit_name=fit_name,
            output_dir=output_dir,
            do_useMINOS=bool(options.get("do_useMINOS", False)),
            params_MINOS=options.get("params_MINOS", []),
            include_real=include_real,
            table_sigfigs=options.get("table_sigfigs", 4),
            refit_fromLocalMinimumMINOS=False,
            show_roofit_output=bool(options.get("show_roofit_output", False)),
            _minos_refit_depth=_minos_refit_depth + 1,
        )
        refit_output["refit_fromLocalMinimumMINOS"] = {
            "triggered": True,
            "parameter": refit_candidate_name,
            "candidate": minos_new_minimum_candidates.get(refit_candidate_name),
            "first_pass_minos_status_by_parameter": dict(minos_status_by_parameter),
            "first_pass_minos_raw_status_by_parameter": dict(minos_raw_status_by_parameter),
        }
        if save_results:
            _save_fit_result(refit_output, output_dir)
        return refit_output
    fit_result = minimizer.save()

    parameters = _current_parameter_values(variables, config)
    parameters["_EF_BiasVoltage_enabled"] = True

    profile = ROOT.trapping_roofit_v5.evaluate_profile(
        _std_vector(x),
        _std_vector([parameters[name] for name in MODEL_PARAMETER_NAMES]),
        int(options.get("steps_per_active_region", 400)),
        int(options.get("n_z_grid", DEFAULT_N_Z_GRID)),
        bool(specs["EF_BiasVoltage"].get("enabled", False)),
        _field_model_kind(config),
    )
    y_fit, y_fit_e, y_fit_h, y_fit_no_offset, y_fit_offset, response = _profile_result_to_dict(profile)
    chi2 = float(np.sum(((y_fit - data) / sigma) ** 2))
    dof = max(data.size - len(fit_names), 1)
    covariance, correlation = _covariance_and_correlation(ROOT, fit_result, fit_names)
    errors = hesse_errors

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(fit_name)).strip("_") or "roofit"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    generation_id = f"{timestamp}_{random.randint(1, 100)}_{safe_name}"
    output = {
        "result": fit_result,
        "minimize_status": status,
        "minos_status": minos_status,
        "minos_names": minos_names,
        "minos_status_by_parameter": minos_status_by_parameter,
        "minos_raw_status_by_parameter": minos_raw_status_by_parameter,
        "minos_new_minimum_candidates": minos_new_minimum_candidates,
        "nll": float(nll.getVal()),
        "generation_id": generation_id,
        "fit_name": safe_name,
        "configuration": config,
        "initial_parameters": _parameter_values(specs),
        "real_parameters": real_parameters,
        "parameters": parameters,
        "fit_names": fit_names,
        "fit_bounds": bounds,
        "covariance": covariance,
        "correlation": correlation,
        "errors": errors,
        "errors_minus": errors_minus,
        "errors_plus": errors_plus,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2 / dof,
        "x": x,
        "y_data": data,
        "y_sigma": sigma,
        "y_sigma_supplied": sigma_supplied,
        "y_fit": y_fit,
        "y_fit_e": y_fit_e,
        "y_fit_h": y_fit_h,
        "y_fit_no_offset": y_fit_no_offset,
        "y_fit_offset": y_fit_offset,
        "profile_offset": float(y_fit_offset),
        "material_response": response,
        "roofit_variables": variables,
        "roofit_nll": nll,
        "roofit_minimizer": minimizer,
        "refit_fromLocalMinimumMINOS": {
            "triggered": False,
            "parameter": None,
            "candidate": None,
        },
        "timings": {
            "minimize_elapsed_s": minimize_elapsed,
            "minos_elapsed_s": minos_elapsed,
            "minos_elapsed_by_parameter_s": minos_elapsed_by_parameter,
            "total_elapsed_s": np.nan,
        },
    }
    if show_summary:
        _print_fit_summary(output)
    if show_fit or show_covariance or show_correlation:
        import matplotlib.pyplot as plt
        if show_fit:
            plot_fit_diagnostics(output)
            plt.show()
        if show_covariance or show_correlation:
            plot_fit_correlation(output)
            plt.show()
    output["timings"]["total_elapsed_s"] = perf_counter() - total_start
    if save_results:
        _save_fit_diagnostic_plot(output, output_dir)
        _save_fit_result(output, output_dir)
    if show_summary:
        print(f"elapsed total fit_trapping_model = {output['timings']['total_elapsed_s']:.3f} s")
    return output


def fit_profile_withTrapps(x_vec, y_data, configuration, y_sigma=None, **kwargs):
    return fit_trapping_model(x_vec, y_data, configuration, y_sigma=y_sigma, **kwargs)


def randomized_initial_configuration(configuration, seed=None, rng=None, include_constrained=False):
    """Return a config copy with random initial values for ranged floating parameters."""
    config = load_fit_configuration(configuration)
    rng = np.random.default_rng(seed) if rng is None else rng
    randomized = copy.deepcopy(config)
    randomized_names = []
    for name, spec in randomized["parameters"].items():
        if spec.get("type") != "fit":
            continue
        if spec.get("constraint") is not None and not include_constrained:
            continue
        if "min" not in spec or "max" not in spec:
            continue
        low = float(spec["min"])
        high = float(spec["max"])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue
        spec["value"] = float(rng.uniform(low, high))
        randomized_names.append(name)
    return randomized, randomized_names


def _text_progress_bar(index, total, start_time, width=32):
    done = int(round(width * index / max(total, 1)))
    elapsed = perf_counter() - start_time
    rate = elapsed / index if index > 0 else np.nan
    remaining = rate * (total - index) if index > 0 else np.nan
    pct = 100.0 * index / max(total, 1)
    eta = "--" if not np.isfinite(remaining) else f"{remaining:.1f}s"
    return f"[{'#' * done}{'.' * (width - done)}] {index}/{total} {pct:5.1f}% ETA {eta}"


def _multifit_summary_record(output, case_index, randomized_parameters):
    return {
        "case_index": int(case_index),
        "generation_id": output.get("generation_id"),
        "fit_name": output.get("fit_name"),
        "minimize_status": output.get("minimize_status"),
        "minos_status": output.get("minos_status"),
        "nll": output.get("nll"),
        "chi2": output.get("chi2"),
        "dof": output.get("dof"),
        "chi2_dof": output.get("chi2_dof"),
        "timings": output.get("timings", {}),
        "randomized_parameters": list(randomized_parameters),
        "initial_parameters": output.get("initial_parameters", {}),
        "final_parameters": output.get("parameters", {}),
        "hesse_errors": {
            name: float(output["errors"][index])
            for index, name in enumerate(output.get("fit_names", []))
        },
        "minos_errors": {
            name: {
                "minus": float(output["errors_minus"][index]),
                "plus": float(output["errors_plus"][index]),
            }
            for index, name in enumerate(output.get("fit_names", []))
        },
    }


def multifit_random_initial_profiles(
    configuration,
    profile_index,
    rootfiles_list=None,
    n_cases=200,
    output_dir="SiC_UVLED/Fits_RooFit/MultiFits",
    seed=None,
    y_sigma=None,
    residual_sigma_fraction=None,
    include_constrained=False,
    fit_name_prefix=None,
    show_current=True,
    show_summary=True,
    show_correlation=False,
    save_results=True,
    **fit_kwargs,
):
    """Run repeated fits with randomized unconstrained ranged initial values."""
    if rootfiles_list is None:
        rootfiles_list = discover_profile_rootfiles("SiC_UVLED/RootFiles")
    xdata, ydata, rootfilename = get_profileData(profile_index, rootfiles_list)
    base_config = load_fit_configuration(configuration)
    options = base_config["fit_options"]
    if y_sigma is None:
        fraction = (
            float(residual_sigma_fraction)
            if residual_sigma_fraction is not None
            else float(options.get("relative_sigma_fraction", 0.02))
        )
        floor_fraction = float(options.get("relative_sigma_floor_fraction", 1e-6))
        sigma_floor = floor_fraction * max(float(np.nanmax(np.abs(ydata))), 1.0)
        y_sigma = fraction * np.maximum(np.abs(ydata), sigma_floor)
    else:
        y_sigma = np.asarray(y_sigma, dtype=float)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "multifit_summary.json"
    rng = np.random.default_rng(seed)
    start_time = perf_counter()
    records = []
    outputs = []

    clear_output = None
    display = None
    if show_current:
        try:
            from IPython.display import clear_output as _clear_output, display as _display
            clear_output = _clear_output
            display = _display
        except Exception:
            clear_output = None
            display = None

    if fit_name_prefix is None:
        fit_name_prefix = f"multifit_profile_{int(profile_index):02d}"

    for case_index in range(int(n_cases)):
        case_config, randomized_names = randomized_initial_configuration(
            base_config,
            rng=rng,
            include_constrained=include_constrained,
        )
        if clear_output is not None:
            clear_output(wait=True)
        print(_text_progress_bar(case_index, int(n_cases), start_time))
        print(f"Current case = {case_index + 1}/{int(n_cases)}")
        print(f"Profile index = {profile_index}")
        print(f"ROOT file = {rootfilename}")
        print(f"Randomized initial parameters = {randomized_names if randomized_names else '--'}")

        output = fit_trapping_model(
            xdata,
            ydata,
            case_config,
            y_sigma=y_sigma,
            show_summary=show_summary,
            show_fit=False,
            show_correlation=show_correlation,
            save_results=save_results,
            fit_name=f"{fit_name_prefix}_{case_index:04d}",
            output_dir=output_path,
            **fit_kwargs,
        )
        record = _multifit_summary_record(output, case_index, randomized_names)
        records.append(record)
        outputs.append(output)
        summary_payload = {
            "profile_index": int(profile_index),
            "rootfilename": str(rootfilename),
            "n_cases_requested": int(n_cases),
            "n_cases_completed": len(records),
            "seed": seed,
            "include_constrained": bool(include_constrained),
            "output_dir": str(output_path),
            "base_configuration": base_config,
            "records": records,
        }
        with summary_path.open("w", encoding="utf-8") as stream:
            json.dump(_json_ready(summary_payload), stream, indent=2, sort_keys=True)
            stream.write("\n")

        if show_current:
            import matplotlib.pyplot as plt
            figure = plot_fit_diagnostics(output)
            if display is not None:
                display(figure)
            else:
                plt.show()
            plt.close(figure)
        print(_text_progress_bar(case_index + 1, int(n_cases), start_time))

    return {
        "profile_index": int(profile_index),
        "rootfilename": str(rootfilename),
        "output_dir": str(output_path),
        "summary_json": str(summary_path),
        "records": records,
        "outputs": outputs,
    }


def load_saved_fit_inputs(json_path):
    return _base.load_saved_fit_inputs(json_path)


def read_fitPars(name_fragments, fits_dir="SiC_UVLED/Fits_RooFit"):
    try:
        return _base.read_fitPars(name_fragments, fits_dir=fits_dir)
    except FileNotFoundError:
        return _base.read_fitPars(name_fragments, fits_dir="SiC_UVLED/Fits")


# ---------------------------------------------------------------------------
# Q+TColl compatibility and diagnostic helpers.
# The RooFit/C++ charge-profile implementation above is the copied baseline.
# ---------------------------------------------------------------------------
try:
    from utils_lib import getVals, wu_rootfile
except Exception:  # pragma: no cover
    getVals = None
    wu_rootfile = None

DEFAULT_TCOLL_THRESHOLDS = [0, 5, 10, 25, 50]
DEFAULT_N_Z_GRID = 20001


PARAMETER_UNITS = {
    "BM_z0": "um",
    "BM_zRight": "um",
    "BM_zR0": "um",
    "BM_z_Aberr": "um",
    "BM_CoefA": "adim.",
    "BM_CoefB": "1/um",
    "BM_area": "NE",
    "BM_scaleAmp": "NE",
    "BM_scaleOffset": "NE",
    "MV_beta_e": "adim.",
    "MV_vsat_e": "um/ns",
    "MV_mu0_e": "um^2/(V ns)",
    "MV_beta_h": "adim.",
    "MV_vsat_h": "um/ns",
    "MV_mu0_h": "um^2/(V ns)",
    "EF_BiasVoltage": "V",
    "EF_CoefA": "V/um^3",
    "EF_CoefB": "V/um^2",
    "EF_CoefC": "V/um",
    "EF_z0": "um",
    "EF_ExpAmpLeft": "V/um",
    "EF_ExpDecayLeft": "um",
    "EF_ExpAmpRight": "V/um",
    "EF_ExpDecayRight": "um",
    "SC_scaleAmp": "adim.",
    "TR_tau_e": "ns",
    "TR_tau_h": "ns",
    "SC_scaleOffset": "NE",
    "SC_scale_zShift": "um",
    "RC_capacitance_pF": "pF",
    "RC_resistance_ohm": "ohm",
    "RC_extra_sigma_ns": "ns",
}


def parameter_label(name: str) -> str:
    return f"{name} / {PARAMETER_UNITS.get(name, 'adim.')}"


def default_configuration() -> dict:
    return {
        "parameters": {
            "BM_z0": {"type": "fixed", "value": -2.561761988852449},
            "BM_zRight": {"type": "fixed", "value": 50.337266961172645},
            "BM_zR0": {"type": "fixed", "value": 4.385374879802671},
            "BM_z_Aberr": {"type": "fixed", "value": 12.624987247362315},
            "BM_CoefA": {"type": "fixed", "value": 0.004461681253026355},
            "BM_CoefB": {"type": "fixed", "value": 0.0017325631823942112},
            "BM_area": {"type": "fixed", "value": 9.106901819568089},
            "BM_scaleAmp": {"type": "fixed", "value": 1.9048051801350496},
            "BM_scaleOffset": {"type": "fixed", "value": -0.031227930105467934},
            "MV_beta_e": {"type": "fixed", "value": 0.96},
            "MV_vsat_e": {"type": "fixed", "value": 147.0},
            "MV_mu0_e": {"type": "fixed", "value": 95.0},
            "MV_beta_h": {"type": "fixed", "value": 1.02},
            "MV_vsat_h": {"type": "fixed", "value": 69.0},
            "MV_mu0_h": {"type": "fixed", "value": 11.5},
            "EF_BiasVoltage": {"enabled": True, "type": "fixed", "value": 850.0},
            "EF_CoefA": {"type": "fixed", "value": 0.0},
            "EF_CoefB": {"type": "fixed", "value": 0.0},
            "EF_CoefC": {"type": "fixed", "value": 0.0},
            "EF_z0": {"type": "fixed", "value": 0.0},
            "EF_ExpAmpLeft": {"type": "fixed", "value": 10.0},
            "EF_ExpDecayLeft": {"type": "fixed", "value": 15.0},
            "EF_ExpAmpRight": {"type": "fixed", "value": 1.0},
            "EF_ExpDecayRight": {"type": "fixed", "value": 15.0},
            "SC_scaleAmp": {"type": "fixed", "value": 1.0},
            "TR_tau_e": {"type": "fixed", "value": 1.0},
            "TR_tau_h": {"type": "fixed", "value": 0.1},
            "SC_scaleOffset": {"type": "fixed", "value": 0.0},
            "SC_scale_zShift": {"type": "fixed", "value": 0.0},
            "RC_capacitance_pF": {"type": "fixed", "value": 16.0},
            "RC_resistance_ohm": {"type": "fixed", "value": 100.0},
            "RC_extra_sigma_ns": {"type": "fixed", "value": 0.0},
        },
        "fit_options": {
            "field_model": "polynomial",
            "steps_per_active_region": 400,
            "n_z_grid": DEFAULT_N_Z_GRID,
            "tColl_threshold_percent": 5.0,
            "waveform_dt_ns": 0.02,
            "waveform_store_indices": [],
            "waveform_store_z_step_um": 1.0,
            "tColl_error_systematic_ns": 0.0,
            "include_tcoll_in_cost": True,
            "ignore_pulse_duration_fit": False,
            "ignore_compute_WFs": False,
            "charge_error_fraction": 0.0175,
            "charge_error_floor": 0.0,
            "pulse_duration_model": "ramo_waveform",
            "rc_model": "first_order_lowpass",
        },
    }


def load_configuration(configuration) -> dict:
    if configuration is None:
        config = default_configuration()
    elif isinstance(configuration, (str, Path)):
        with Path(configuration).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        config = payload.get("configuration", payload)
    elif isinstance(configuration, dict):
        config = copy.deepcopy(configuration.get("configuration", configuration))
    else:
        raise TypeError("configuration must be None, a dict, or a JSON file path")

    defaults = default_configuration()
    config.setdefault("parameters", {})
    config.setdefault("fit_options", {})
    for name, spec in defaults["parameters"].items():
        config["parameters"].setdefault(name, copy.deepcopy(spec))
        config["parameters"][name].setdefault("type", spec["type"])
        config["parameters"][name].setdefault("value", spec["value"])
    for name, value in defaults["fit_options"].items():
        config["fit_options"].setdefault(name, value)
    config["fit_options"]["field_model"] = canonical_field_model(config["fit_options"].get("field_model", "polynomial"))
    return config


def parameter_values(config: dict) -> dict:
    cfg = load_configuration(config)
    values = {name: float(spec["value"]) for name, spec in cfg["parameters"].items()}
    values["_EF_BiasVoltage_enabled"] = bool(cfg["parameters"]["EF_BiasVoltage"].get("enabled", False))
    apply_derived_field_values(values, cfg)
    return values


def canonical_field_model(field_model) -> str:
    text = str(field_model).strip().lower().replace("-", "_")
    aliases = {
        "poly": "polynomial",
        "polynomial": "polynomial",
        "double_exp": "double_exponential",
        "double_exponential": "double_exponential",
        "doublejunction": "double_exponential",
        "double_junction": "double_exponential",
    }
    if text not in aliases:
        raise ValueError("field_model must be 'polynomial' or 'double_exponential'")
    return aliases[text]


def exp_integral(width: float, decay: float) -> float:
    width = float(width)
    decay = float(decay)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("active width must be finite and positive")
    if not np.isfinite(decay) or decay <= 0.0:
        raise ValueError("double-exponential decay lengths must be finite and positive")
    return decay * (1.0 - np.exp(-width / decay))


def derived_field_constant(values: dict) -> float:
    width = float(values["BM_zRight"])
    center = float(values["EF_z0"])
    u_left = -center
    u_right = width - center
    quadratic_integral = float(values["EF_CoefA"]) * (u_right**3 - u_left**3) / 3.0
    linear_integral = float(values["EF_CoefB"]) * (u_right**2 - u_left**2) / 2.0
    return (float(values["EF_BiasVoltage"]) - quadratic_integral - linear_integral) / width


def derived_exp_amp_right(values: dict) -> float:
    width = float(values["BM_zRight"])
    left_integral = exp_integral(width, values["EF_ExpDecayLeft"])
    right_integral = exp_integral(width, values["EF_ExpDecayRight"])
    return (float(values["EF_BiasVoltage"]) - float(values["EF_ExpAmpLeft"]) * left_integral) / right_integral


def apply_derived_field_values(values: dict, config: dict) -> dict:
    if not bool(config["parameters"]["EF_BiasVoltage"].get("enabled", False)):
        return values
    if canonical_field_model(config["fit_options"].get("field_model", "polynomial")) == "double_exponential":
        values["EF_ExpAmpRight"] = derived_exp_amp_right(values)
    else:
        values["EF_CoefC"] = derived_field_constant(values)
    return values


def discover_profile_rootfiles(root_dir="SiC_UVLED/RootFiles") -> list[str]:
    return sorted(str(path) for path in Path(root_dir).glob("*.root"))


def _require_root_helpers():
    if getVals is None:
        raise RuntimeError("utils_lib.getVals is not available in this environment")


def profile_normalization_from_1mw2(rootfiles_list) -> float:
    _require_root_helpers()
    matches = [rootfile for rootfile in rootfiles_list if "1MW2" in Path(rootfile).name]
    if not matches:
        raise FileNotFoundError("No 1MW2 ROOT file found for profile normalization")
    vals = getVals(matches[0], ["Profiles:ChargeCSPA_Avr"])
    norm = float(np.nanmax(np.abs(np.asarray(vals["Profiles:ChargeCSPA_Avr"], dtype=float))))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Invalid 1MW2 normalization")
    return norm


def get_profile_data(index: int, rootfiles_list, normalize_to_1mw2=True):
    _require_root_helpers()
    rootfiles = list(rootfiles_list)
    rootfilename = str(rootfiles[int(index)])
    vals = getVals(rootfilename, ["Profiles:zSiC", "Profiles:ChargeCSPA_Avr"])
    x = np.asarray(vals["Profiles:zSiC"], dtype=float).ravel()
    y = np.abs(np.asarray(vals["Profiles:ChargeCSPA_Avr"], dtype=float).ravel())
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if normalize_to_1mw2:
        y = y / profile_normalization_from_1mw2(rootfiles)
    return x, y, rootfilename


def _profile_error_arrays(rootfilename, tree_name="Charge", error_branch="TPA_E"):
    branch_spec = f"{tree_name}:{error_branch}"
    vals = getVals(rootfilename, ["Profiles:zSiC", branch_spec])
    x = np.asarray(vals["Profiles:zSiC"], dtype=float).ravel()
    yerr = np.abs(np.asarray(vals[branch_spec], dtype=float).ravel())
    n = min(x.size, yerr.size)
    x, yerr = x[:n], yerr[:n]
    finite = np.isfinite(x) & np.isfinite(yerr)
    x, yerr = x[finite], yerr[finite]
    order = np.argsort(x)
    return x[order], yerr[order]


def get_profile_data_with_errors(
    index: int,
    rootfiles_list,
    normalize_to_1mw2=True,
    systematic_error=0.0,
    fallback_fraction=0.0175,
    error_branch_candidates=None,
):
    x, y, rootfilename = get_profile_data(index, rootfiles_list, normalize_to_1mw2=normalize_to_1mw2)
    fallback = constant_sigma_from_profile(y, fraction=fallback_fraction)
    candidates = error_branch_candidates or [
        ("Charge", "TPA_best_masked_E"),
        ("Charge", "TPA_best_E"),
        ("Charge", "TPA_E"),
        ("Profiles", "ChargeCSPA_TPA_best_Avr_E"),
        ("Profiles", "ChargeCSPA_Avr_E"),
        ("Profiles", "Charge_Avr_E"),
        ("Profiles", "ChargeCSPA_E"),
        ("Profiles", "Charge_E"),
    ]

    source = None
    yerr = None
    for tree_name, branch_name in candidates:
        try:
            xerr, raw_err = _profile_error_arrays(rootfilename, tree_name, branch_name)
        except Exception:
            continue
        if raw_err.size == 0:
            continue
        source = f"{tree_name}:{branch_name}"
        norm = profile_normalization_from_1mw2(rootfiles_list) if normalize_to_1mw2 else 1.0
        yerr = raw_err / norm
        if xerr.shape != x.shape or not np.allclose(xerr, x, rtol=0.0, atol=1e-9):
            yerr = np.interp(x, xerr, yerr, left=np.nan, right=np.nan)
        break

    if yerr is None:
        source = "constant fallback"
        yerr = fallback
    else:
        yerr = np.asarray(yerr, dtype=float)
        invalid = (~np.isfinite(yerr)) | (yerr <= 0.0)
        if np.any(invalid):
            yerr[invalid] = fallback[invalid]

    syst = float(systematic_error)
    if np.isfinite(syst) and syst > 0.0:
        yerr = np.sqrt(yerr**2 + syst**2)
    return x, y, yerr, rootfilename, source


def average_tcoll_repetitions_from_root(rootfilename, source_tree="SignalTPA_best_masked", dz_sic=2.83, z_offset_um=-38.0):
    _require_root_helpers()
    vals = getVals(rootfilename, [f"{source_tree}:tColl", "Raw:reps", "Raw:z"])
    tcoll = np.asarray(vals[f"{source_tree}:tColl"], dtype=float)
    reps = int(vals["Raw:reps"])
    z_raw = np.asarray(vals["Raw:z"], dtype=float).ravel()
    if tcoll.ndim != 2:
        raise ValueError(f"{source_tree}:tColl must be 2D, got shape {tcoll.shape}")
    if reps <= 0:
        raise ValueError(f"Invalid Raw:reps={reps}")
    n_z = min(int(tcoll.shape[1]) // reps, z_raw.size)
    if n_z <= 0:
        raise ValueError(f"Cannot infer z points from tColl shape={tcoll.shape}, reps={reps}")
    usable = reps * n_z
    tcoll_rep = tcoll[:, :usable].reshape(tcoll.shape[0], reps, n_z)
    return {
        "z_sic": z_raw[:n_z] * float(dz_sic) + float(z_offset_um),
        "thresholds": np.asarray(DEFAULT_TCOLL_THRESHOLDS[:tcoll.shape[0]], dtype=float),
        "tColl_Avr": np.nanmean(tcoll_rep, axis=1),
        "tColl_Err": np.nanstd(tcoll_rep, axis=1),
    }


def get_tcoll_data(index: int, rootfiles_list, threshold_percent=5, source_tree="SignalTPA_best_masked", error_floor_ns=0.0):
    _require_root_helpers()
    rootfiles = list(rootfiles_list)
    rootfilename = str(rootfiles[int(index)])
    specs = [
        f"{source_tree}:tColl_Avr",
        f"{source_tree}:tColl_Err",
        f"{source_tree}:tColl_zSiC",
        f"{source_tree}:tColl_thresholds",
    ]
    try:
        vals = getVals(rootfilename, specs)
        tcoll = np.asarray(vals[f"{source_tree}:tColl_Avr"], dtype=float)
        tcoll_err = np.asarray(vals[f"{source_tree}:tColl_Err"], dtype=float)
        z = np.asarray(vals[f"{source_tree}:tColl_zSiC"], dtype=float).ravel()
        thresholds = np.asarray(vals[f"{source_tree}:tColl_thresholds"], dtype=float).ravel()
    except Exception:
        averaged = average_tcoll_repetitions_from_root(rootfilename, source_tree=source_tree)
        tcoll = averaged["tColl_Avr"]
        tcoll_err = averaged["tColl_Err"]
        z = averaged["z_sic"]
        thresholds = averaged["thresholds"]
    ith = int(np.argmin(np.abs(thresholds - float(threshold_percent))))
    yerr = np.asarray(tcoll_err[ith], dtype=float)
    floor = float(error_floor_ns)
    if np.isfinite(floor) and floor > 0.0:
        yerr = np.sqrt(yerr**2 + floor**2)
    return z, tcoll[ith], yerr, rootfilename, float(thresholds[ith])


def zR_aberracion_esferica(z, zR0, z0, a, b):
    z = np.asarray(z, dtype=float)
    dz = np.maximum(z - z0, 0.0)
    return np.maximum(float(zR0) + float(a) * dz + float(b) * dz * dz, 1e-12)


def trapezoid(y, x):
    return np.trapz(y, x)


def interp_uniform(y, z_min, dz, z):
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    pos = (z - z_min) / dz
    idx = np.floor(pos).astype(int)
    idx = np.clip(idx, 0, y.size - 2)
    frac = pos - idx
    return y[idx] * (1.0 - frac) + y[idx + 1] * frac


def electric_field(z_active, values, config):
    z_active = np.asarray(z_active, dtype=float)
    width = float(values["BM_zRight"])
    z_left = float(values["BM_z0"]) + float(values["SC_scale_zShift"])
    model = canonical_field_model(config["fit_options"].get("field_model", "polynomial"))
    if model == "double_exponential":
        s = z_active - z_left
        lam_l = max(float(values["EF_ExpDecayLeft"]), 1e-12)
        lam_r = max(float(values["EF_ExpDecayRight"]), 1e-12)
        efield = values["EF_ExpAmpLeft"] * np.exp(-s / lam_l) + values["EF_ExpAmpRight"] * np.exp(-(width - s) / lam_r)
    else:
        center_abs = z_left + float(values["EF_z0"])
        u = z_active - center_abs
        efield = values["EF_CoefA"] * u * u + values["EF_CoefB"] * u + values["EF_CoefC"]
    return np.maximum(efield, 0.0)


def mobility_from_field(field, mu0, vsat, beta):
    field = np.asarray(field, dtype=float)
    out = np.zeros_like(field)
    valid = field > 1e-2
    out[valid] = float(mu0) / np.power(
        1.0 + np.power(float(mu0) * field[valid] / float(vsat), float(beta)),
        1.0 / float(beta),
    )
    return out


def cumulative_trapz_from_left(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y)
    if y.size < 2:
        return out
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
    return out


def cumulative_trapz_from_right(y, x):
    return cumulative_trapz_from_left(y[::-1], x[::-1])[::-1] * -1.0


def roofit_response_from_velocities(z, v_e, v_h, tau_e, tau_h):
    z = np.asarray(z, dtype=float)
    v_e = np.asarray(v_e, dtype=float)
    v_h = np.asarray(v_h, dtype=float)
    steps = z.size
    if steps < 2:
        return np.zeros_like(z), np.zeros_like(z)
    width = float(z[-1] - z[0])
    dz = width / float(steps - 1)
    response_e = np.zeros_like(z)
    response_h = np.zeros_like(z)
    survival_e = np.zeros(steps - 1, dtype=float)
    survival_h = np.zeros(steps - 1, dtype=float)
    response_step_e = np.zeros(steps - 1, dtype=float)
    response_step_h = np.zeros(steps - 1, dtype=float)

    for i in range(steps - 1):
        active_e = (
            v_e[i] > 0.0 and v_e[i + 1] > 0.0
            and np.isfinite(v_e[i]) and np.isfinite(v_e[i + 1])
        )
        active_h = (
            v_h[i] > 0.0 and v_h[i + 1] > 0.0
            and np.isfinite(v_h[i]) and np.isfinite(v_h[i + 1])
        )
        if active_e and np.isfinite(tau_e) and tau_e > 0.0:
            inverse_velocity = 0.5 * (1.0 / v_e[i] + 1.0 / v_e[i + 1])
            survival_e[i] = np.exp(-(dz * inverse_velocity) / tau_e)
            response_step_e[i] = dz / width
        if active_h and np.isfinite(tau_h) and tau_h > 0.0:
            inverse_velocity = 0.5 * (1.0 / v_h[i] + 1.0 / v_h[i + 1])
            survival_h[i] = np.exp(-(dz * inverse_velocity) / tau_h)
            response_step_h[i] = dz / width

    for i in range(steps - 2, -1, -1):
        response_e[i] = response_step_e[i] + survival_e[i] * response_e[i + 1]
    for i in range(1, steps):
        response_h[i] = response_step_h[i - 1] + survival_h[i - 1] * response_h[i - 1]
    return response_e, response_h


def build_response_maps(config):
    cfg = load_configuration(config)
    values = parameter_values(cfg)
    steps = max(int(cfg["fit_options"].get("steps_per_active_region", 400)), 2)
    z_left = values["BM_z0"] + values["SC_scale_zShift"]
    width = values["BM_zRight"]
    z = z_left + np.linspace(0.0, width, steps)
    dz = width / (steps - 1)
    field = electric_field(z, values, cfg)

    mu_e = mobility_from_field(field, values["MV_mu0_e"], values["MV_vsat_e"], values["MV_beta_e"])
    mu_h = mobility_from_field(field, values["MV_mu0_h"], values["MV_vsat_h"], values["MV_beta_h"])
    v_e = mu_e * field
    v_h = mu_h * field

    inv_v_e = np.full_like(v_e, np.inf)
    inv_v_h = np.full_like(v_h, np.inf)
    valid_e = v_e > 0.0
    valid_h = v_h > 0.0
    inv_v_e[valid_e] = 1.0 / v_e[valid_e]
    inv_v_h[valid_h] = 1.0 / v_h[valid_h]

    drift_time_e = cumulative_trapz_from_right(inv_v_e, z)
    drift_time_h = cumulative_trapz_from_left(inv_v_h, z)

    tau_e = float(values["TR_tau_e"])
    tau_h = float(values["TR_tau_h"])
    response_e, response_h = roofit_response_from_velocities(z, v_e, v_h, tau_e, tau_h)
    response_total = response_e + response_h

    dominant = np.where(drift_time_e >= drift_time_h, "e", "h")
    pulse_intrinsic = np.maximum(drift_time_e, drift_time_h)
    rc_tau_ns = rc_tau_ns_from_values(values)
    pulse_e_rc = rc_broaden_duration(drift_time_e, rc_tau_ns, values["RC_extra_sigma_ns"])
    pulse_h_rc = rc_broaden_duration(drift_time_h, rc_tau_ns, values["RC_extra_sigma_ns"])
    pulse_rc = rc_broaden_duration(pulse_intrinsic, rc_tau_ns, values["RC_extra_sigma_ns"])

    return {
        "z": z,
        "dz": dz,
        "efield": field,
        "mobility_e": mu_e,
        "mobility_h": mu_h,
        "vdrift_e": v_e,
        "vdrift_h": v_h,
        "drift_time_e": drift_time_e,
        "drift_time_h": drift_time_h,
        "response_e": response_e,
        "response_h": response_h,
        "response_total": response_total,
        "dominant_carrier": dominant,
        "pulse_duration_e_intrinsic": drift_time_e,
        "pulse_duration_h_intrinsic": drift_time_h,
        "pulse_duration_intrinsic": pulse_intrinsic,
        "pulse_duration_e_rc": pulse_e_rc,
        "pulse_duration_h_rc": pulse_h_rc,
        "pulse_duration_rc": pulse_rc,
        "rc_tau_ns": rc_tau_ns,
    }


def rc_tau_ns_from_values(values):
    # ohm*pF = ps, hence /1000 gives ns.
    return float(values["RC_resistance_ohm"]) * float(values["RC_capacitance_pF"]) / 1000.0


def rc_broaden_duration(duration_ns, rc_tau_ns, extra_sigma_ns=0.0):
    duration_ns = np.asarray(duration_ns, dtype=float)
    broad = np.sqrt(np.maximum(duration_ns, 0.0) ** 2 + float(rc_tau_ns) ** 2 + float(extra_sigma_ns) ** 2)
    broad[~np.isfinite(duration_ns)] = np.inf
    return broad


def rc_lowpass_filter(waveform, dt_ns, tau_ns):
    waveform = np.asarray(waveform, dtype=float)
    tau_ns = float(tau_ns)
    if not np.isfinite(tau_ns) or tau_ns <= 0.0:
        return waveform.copy()
    alpha = np.exp(-float(dt_ns) / tau_ns)
    out = np.zeros_like(waveform)
    if waveform.size == 0:
        return out
    out[0] = (1.0 - alpha) * waveform[0]
    for i in range(1, waveform.size):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * waveform[i]
    return out


def threshold_duration(time_ns, waveform, threshold_percent=5.0):
    time_ns = np.asarray(time_ns, dtype=float)
    waveform = np.asarray(waveform, dtype=float)
    if time_ns.size >= 2 and waveform.size == time_ns.size:
        dt = float(np.nanmedian(np.diff(time_ns)))
        if np.isfinite(dt) and dt > 0.0:
            time_ns = np.r_[time_ns[0] - dt, time_ns]
            waveform = np.r_[0.0, waveform]
    if time_ns.shape != waveform.shape or waveform.size < 3:
        return np.nan
    finite = np.isfinite(time_ns) & np.isfinite(waveform)
    if not np.any(finite):
        return np.nan
    y = np.where(finite, waveform, np.nan)
    peak_index = int(np.nanargmax(y))
    peak = float(y[peak_index])
    if not np.isfinite(peak) or peak <= 0.0:
        return np.nan
    level = peak * float(threshold_percent) / 100.0

    left = y[: peak_index + 1]
    left_cross = np.where(left <= level)[0]
    if left_cross.size:
        i0 = int(left_cross[-1])
        i1 = min(i0 + 1, peak_index)
    else:
        i0 = 0
        i1 = min(1, peak_index)

    right = y[peak_index:]
    right_cross = np.where(right <= level)[0]
    if right_cross.size:
        j1 = peak_index + int(right_cross[0])
        j0 = max(peak_index, j1 - 1)
    else:
        j1 = waveform.size - 1
        j0 = max(peak_index, j1 - 1)

    def interp_cross(i_low, i_high):
        y0, y1 = y[i_low], y[i_high]
        t0, t1 = time_ns[i_low], time_ns[i_high]
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
            return float(t0)
        frac = (level - y0) / (y1 - y0)
        return float(t0 + frac * (t1 - t0))

    t_left = interp_cross(i0, i1)
    t_right = interp_cross(j0, j1)
    return max(t_right - t_left, 0.0)


def fall_duration_from_t0(time_ns, waveform, threshold_percent=5.0):
    time_ns = np.asarray(time_ns, dtype=float)
    waveform = np.asarray(waveform, dtype=float)
    if time_ns.shape != waveform.shape or waveform.size < 3:
        return np.nan
    finite = np.isfinite(time_ns) & np.isfinite(waveform)
    if not np.any(finite):
        return np.nan
    y = np.where(finite, waveform, np.nan)
    peak_index = int(np.nanargmax(y))
    peak = float(y[peak_index])
    if not np.isfinite(peak) or peak <= 0.0:
        return np.nan
    level = peak * float(threshold_percent) / 100.0
    right = y[peak_index:]
    cross = np.where(right <= level)[0]
    if not cross.size:
        return float(time_ns[-1] - time_ns[0])
    j1 = peak_index + int(cross[0])
    j0 = max(peak_index, j1 - 1)
    y0, y1 = y[j0], y[j1]
    t0, t1 = time_ns[j0], time_ns[j1]
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
        t_fall = float(t1)
    else:
        t_fall = float(t0 + (level - y0) * (t1 - t0) / (y1 - y0))
    return max(t_fall - float(time_ns[0]), 0.0)


def _finite_max_or(values, fallback):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(fallback)
    return float(np.nanmax(finite))


def waveform_store_indices_auto(x_values, charge_no_offset, z_min=0.0, z_max=50.0, step_um=1.0, min_charge_fraction=0.0):
    x_values = np.asarray(x_values, dtype=float)
    charge_no_offset = np.asarray(charge_no_offset, dtype=float)
    if x_values.size == 0:
        return []
    finite_charge = charge_no_offset[np.isfinite(charge_no_offset)]
    if finite_charge.size == 0:
        charge_threshold = -np.inf
    else:
        charge_threshold = float(np.nanmax(finite_charge)) * float(min_charge_fraction)
    active_candidates = np.flatnonzero(
        np.isfinite(x_values)
        & np.isfinite(charge_no_offset)
        & (charge_no_offset >= charge_threshold)
        & (x_values >= float(z_min))
        & (x_values <= float(z_max))
    )
    if active_candidates.size == 0:
        active_candidates = np.flatnonzero(np.isfinite(x_values) & np.isfinite(charge_no_offset) & (charge_no_offset >= charge_threshold))
    if active_candidates.size == 0:
        return []
    step_um = float(step_um)
    if not np.isfinite(step_um) or step_um <= 0.0:
        step_um = 1.0
    target_z = np.arange(
        max(float(z_min), float(np.nanmin(x_values[active_candidates]))),
        min(float(z_max), float(np.nanmax(x_values[active_candidates]))) + 0.5 * step_um,
        step_um,
    )
    selected = []
    for z_target in target_z:
        distances = np.abs(x_values[active_candidates] - z_target)
        selected.append(int(active_candidates[int(np.nanargmin(distances))]))
    return sorted(set(selected))


def waveform_time_axis(response, dt_ns, rc_tau_ns):
    max_time = max(
        _finite_max_or(response["drift_time_e"], 0.0),
        _finite_max_or(response["drift_time_h"], 0.0),
        1.0,
    )
    t_max = max_time + 8.0 * max(float(rc_tau_ns), 0.0) + 5.0 * float(dt_ns)
    n = max(int(np.ceil(t_max / float(dt_ns))) + 1, 8)
    return np.arange(n, dtype=float) * float(dt_ns)


def induced_waveforms_from_generation(generated, response, width, tau_e, tau_h, time_ns):
    generated = np.asarray(generated, dtype=float)
    z = np.asarray(response["z"], dtype=float)
    v_e = np.asarray(response["vdrift_e"], dtype=float)
    v_h = np.asarray(response["vdrift_h"], dtype=float)
    dt_ns = float(time_ns[1] - time_ns[0]) if time_ns.size > 1 else 1.0
    dz = float(width) / float(max(z.size - 1, 1))
    q_weights = generated * dz
    q_weights[0] *= 0.5
    q_weights[-1] *= 0.5
    nseg = z.size - 1
    wf_e = np.zeros_like(time_ns)
    wf_h = np.zeros_like(time_ns)

    for start in range(z.size):
        weight = q_weights[start]
        if not np.isfinite(weight) or weight <= 0.0:
            continue

        # Electrons drift towards the right side.
        t_elapsed = 0.0
        survival = 1.0
        for seg in range(start, nseg):
            v_mid = 0.5 * (v_e[seg] + v_e[seg + 1])
            if not np.isfinite(v_mid) or v_mid <= 0.0:
                break
            dt_seg = dz / v_mid
            t_mid = t_elapsed + 0.5 * dt_seg
            index = int(np.floor(t_mid / dt_ns))
            if 0 <= index < wf_e.size:
                wf_e[index] += weight * survival * v_mid / float(width)
            if np.isfinite(tau_e) and tau_e > 0.0:
                survival *= np.exp(-dt_seg / tau_e)
            else:
                survival = 0.0
            t_elapsed += dt_seg
            if survival <= 0.0:
                break

        # Holes drift towards the left side.
        t_elapsed = 0.0
        survival = 1.0
        for seg in range(start - 1, -1, -1):
            v_mid = 0.5 * (v_h[seg] + v_h[seg + 1])
            if not np.isfinite(v_mid) or v_mid <= 0.0:
                break
            dt_seg = dz / v_mid
            t_mid = t_elapsed + 0.5 * dt_seg
            index = int(np.floor(t_mid / dt_ns))
            if 0 <= index < wf_h.size:
                wf_h[index] += weight * survival * v_mid / float(width)
            if np.isfinite(tau_h) and tau_h > 0.0:
                survival *= np.exp(-dt_seg / tau_h)
            else:
                survival = 0.0
            t_elapsed += dt_seg
            if survival <= 0.0:
                break
    return wf_e, wf_h


def simulate_q_tcoll_model(x_vec, configuration):
    cfg = load_configuration(configuration)
    values = parameter_values(cfg)
    response = build_response_maps(cfg)
    z_active = response["z"]
    width = values["BM_zRight"]
    rc_tau = rc_tau_ns_from_values(values)
    waveform_dt = float(cfg["fit_options"].get("waveform_dt_ns", 0.02))
    threshold_percent = float(cfg["fit_options"].get("tColl_threshold_percent", 5.0))
    ignore_compute_wfs = bool(cfg["fit_options"].get("ignore_compute_WFs", False))
    if ignore_compute_wfs:
        cfg["fit_options"]["ignore_pulse_duration_fit"] = True
        cfg["fit_options"]["include_tcoll_in_cost"] = False
    time_axis = np.asarray([], dtype=float) if ignore_compute_wfs else waveform_time_axis(response, waveform_dt, rc_tau)
    store_indices = cfg["fit_options"].get("waveform_store_indices", [])
    do_auto_store_waveforms = isinstance(store_indices, str) and store_indices.lower() == "auto"

    x = np.asarray(x_vec, dtype=float).ravel()
    if not do_auto_store_waveforms:
        store_indices = list(store_indices) if isinstance(store_indices, (list, tuple)) else []
    nz = max(int(cfg["fit_options"].get("n_z_grid", DEFAULT_N_Z_GRID)), 2)
    z_grid_min = float(np.nanmin(x)) - 100.0
    z_grid_max = float(np.nanmax(x)) + 500.0
    z_grid = np.linspace(z_grid_min, z_grid_max, nz)
    dz_grid = (z_grid_max - z_grid_min) / (nz - 1)

    charge_e = np.zeros_like(x)
    charge_h = np.zeros_like(x)
    fill_tcoll = 1.0 if ignore_compute_wfs else np.nan
    tcoll_intrinsic = np.full_like(x, fill_tcoll)
    tcoll_rc = np.full_like(x, fill_tcoll)
    tcoll_e_intrinsic = np.full_like(x, fill_tcoll)
    tcoll_h_intrinsic = np.full_like(x, fill_tcoll)
    tcoll_e_rc = np.full_like(x, fill_tcoll)
    tcoll_h_rc = np.full_like(x, fill_tcoll)
    dominant_at_x = np.empty(x.shape, dtype="U1")
    stored_waveforms = {}
    candidate_waveforms = {}

    amplitude = np.empty_like(z_grid)
    density = np.empty_like(z_grid)
    for ix, zc in enumerate(x):
        zr = zR_aberracion_esferica(zc, values["BM_zR0"], values["BM_z_Aberr"] + values["SC_scale_zShift"], values["BM_CoefA"], values["BM_CoefB"])
        scaled = (z_grid - zc) / zr
        intensity = 1.0 / (1.0 + scaled * scaled)
        amplitude[:] = np.sqrt(np.maximum(intensity, 1e-15))
        area = trapezoid(amplitude, z_grid)
        norm = values["BM_area"] / area
        density[:] = (norm * amplitude) ** 2
        generated = interp_uniform(density, z_grid_min, dz_grid, z_active) * values["SC_scaleAmp"]

        e_density = generated * response["response_e"]
        h_density = generated * response["response_h"]
        e_int = trapezoid(e_density, z_active)
        h_int = trapezoid(h_density, z_active)
        charge_e[ix] = values["BM_scaleAmp"] * e_int
        charge_h[ix] = values["BM_scaleAmp"] * h_int

        total_weight = e_density + h_density
        if ignore_compute_wfs:
            dominant_at_x[ix] = "-"
        elif np.nansum(total_weight) > 0.0:
            wf_e, wf_h = induced_waveforms_from_generation(
                generated,
                response,
                width,
                float(values["TR_tau_e"]),
                float(values["TR_tau_h"]),
                time_axis,
            )
            wf_total = wf_e + wf_h
            wf_e_rc = rc_lowpass_filter(wf_e, waveform_dt, rc_tau)
            wf_h_rc = rc_lowpass_filter(wf_h, waveform_dt, rc_tau)
            wf_total_rc = wf_e_rc + wf_h_rc
            tcoll_e_intrinsic[ix] = threshold_duration(time_axis, wf_e, threshold_percent)
            tcoll_h_intrinsic[ix] = threshold_duration(time_axis, wf_h, threshold_percent)
            tcoll_intrinsic[ix] = threshold_duration(time_axis, wf_total, threshold_percent)
            tcoll_e_rc[ix] = threshold_duration(time_axis, wf_e_rc, threshold_percent)
            tcoll_h_rc[ix] = threshold_duration(time_axis, wf_h_rc, threshold_percent)
            tcoll_rc[ix] = threshold_duration(time_axis, wf_total_rc, threshold_percent)
            dominant_at_x[ix] = "e" if np.nanmax(wf_e_rc) >= np.nanmax(wf_h_rc) else "h"
            if do_auto_store_waveforms:
                should_store = True
            else:
                requested_indices = [
                    (x.size // 2 if item is None else (x.size + int(item) if int(item) < 0 else int(item)))
                    for item in store_indices
                ]
                should_store = ix in requested_indices
            if should_store:
                candidate_waveforms[int(ix)] = {
                    "x": float(zc),
                    "time_ns": time_axis.copy(),
                    "e": wf_e,
                    "h": wf_h,
                    "total": wf_total,
                    "e_rc": wf_e_rc,
                    "h_rc": wf_h_rc,
                    "total_rc": wf_total_rc,
                    "threshold_percent": threshold_percent,
                }
        else:
            dominant_at_x[ix] = "-"

    charge_no_offset = charge_e + charge_h
    if do_auto_store_waveforms:
        selected_indices = waveform_store_indices_auto(
            x,
            charge_no_offset,
            z_min=0.0,
            z_max=width,
            step_um=float(cfg["fit_options"].get("waveform_store_z_step_um", 1.0)),
        )
        stored_waveforms = {
            index: candidate_waveforms[index]
            for index in selected_indices
            if index in candidate_waveforms
        }
    else:
        stored_waveforms = candidate_waveforms
    offset = values["BM_scaleOffset"] + values["SC_scaleOffset"]
    charge_total = charge_no_offset + offset
    return {
        "x": x,
        "charge_total": charge_total,
        "charge_e": charge_e,
        "charge_h": charge_h,
        "charge_no_offset": charge_no_offset,
        "charge_offset": offset,
        "tcoll_intrinsic": tcoll_intrinsic,
        "tcoll_rc": tcoll_rc,
        "tcoll_e_intrinsic": tcoll_e_intrinsic,
        "tcoll_h_intrinsic": tcoll_h_intrinsic,
        "tcoll_e_rc": tcoll_e_rc,
        "tcoll_h_rc": tcoll_h_rc,
        "dominant_carrier_x": dominant_at_x,
        "waveforms": stored_waveforms,
        "response": response,
        "configuration": cfg,
        "parameters": values,
    }


def constant_sigma_from_profile(profile, fraction=0.0175, floor=0.0):
    profile = np.asarray(profile, dtype=float)
    scale = float(np.nanmax(np.abs(profile))) if profile.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    sigma = np.full(profile.shape, float(fraction) * scale, dtype=float)
    floor = float(floor)
    if np.isfinite(floor) and floor > 0.0:
        sigma = np.sqrt(sigma**2 + floor**2)
    return sigma


def gaussian_nll_chi2(data, model, sigma):
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = (
        np.isfinite(data)
        & np.isfinite(model)
        & np.isfinite(sigma)
        & (sigma > 0.0)
    )
    if not np.any(valid):
        return {
            "n": 0,
            "chi2": np.nan,
            "nll": np.nan,
            "residual": np.asarray([], dtype=float),
            "pull": np.asarray([], dtype=float),
        }
    residual = data[valid] - model[valid]
    pull = residual / sigma[valid]
    chi2 = float(np.sum(pull**2))
    nll = float(np.sum(0.5 * pull**2 + np.log(sigma[valid]) + 0.9189385332046727))
    return {
        "n": int(np.sum(valid)),
        "chi2": chi2,
        "nll": nll,
        "residual": residual,
        "pull": pull,
    }


def compute_q_tcoll_metrics(result, data=None, include_tcoll_in_cost=None):
    cfg = result.get("configuration", {})
    options = cfg.get("fit_options", {})
    if include_tcoll_in_cost is None:
        include_tcoll_in_cost = bool(options.get("include_tcoll_in_cost", True))
    if bool(options.get("ignore_pulse_duration_fit", False)) or bool(options.get("ignore_compute_WFs", False)):
        include_tcoll_in_cost = False
    metrics = {
        "include_tcoll_in_cost": bool(include_tcoll_in_cost),
        "charge": None,
        "tcoll": None,
        "combined": None,
    }
    if data is None:
        return metrics

    if "x_q" in data and "y_q" in data:
        x_q = np.asarray(data["x_q"], dtype=float)
        y_q = np.asarray(data["y_q"], dtype=float)
        q_model = np.interp(x_q, result["x"], result["charge_total"], left=np.nan, right=np.nan)
        y_q_sigma = data.get("y_q_err")
        if y_q_sigma is None:
            y_q_sigma = constant_sigma_from_profile(
                y_q,
                fraction=float(options.get("charge_error_fraction", 0.0175)),
                floor=float(options.get("charge_error_floor", 0.0)),
            )
        metrics["charge"] = gaussian_nll_chi2(y_q, q_model, y_q_sigma)

    if "x_t" in data and "y_t" in data:
        x_t = np.asarray(data["x_t"], dtype=float)
        y_t = np.asarray(data["y_t"], dtype=float)
        t_model = np.interp(x_t, result["x"], result["tcoll_rc"], left=np.nan, right=np.nan)
        y_t_sigma = data.get("y_t_err")
        if y_t_sigma is None:
            y_t_sigma = np.ones_like(y_t, dtype=float)
        metrics["tcoll"] = gaussian_nll_chi2(y_t, t_model, y_t_sigma)

    combined_nll = 0.0
    combined_chi2 = 0.0
    combined_n = 0
    for key, enabled in (("charge", True), ("tcoll", bool(include_tcoll_in_cost))):
        part = metrics.get(key)
        if enabled and part is not None and part["n"] > 0:
            combined_nll += part["nll"]
            combined_chi2 += part["chi2"]
            combined_n += part["n"]
    metrics["combined"] = {
        "n": combined_n,
        "chi2": float(combined_chi2) if combined_n else np.nan,
        "nll": float(combined_nll) if combined_n else np.nan,
        "chi2_dof": float(combined_chi2 / max(combined_n, 1)) if combined_n else np.nan,
    }
    return metrics


def _metric_text(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{value:.4g}" if np.isfinite(value) else "--"


def plot_q_tcoll_diagnostics(result, data=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 0.55, 1.0], hspace=0.55, wspace=0.38)
    ax_q = fig.add_subplot(grid[0:2, 0:2])
    ax_q_res = fig.add_subplot(grid[2, 0:2], sharex=ax_q)
    ax_t = fig.add_subplot(grid[0:2, 2:4])
    ax_t_res = fig.add_subplot(grid[2, 2:4], sharex=ax_t)
    ax_field = fig.add_subplot(grid[3, 0])
    ax_v = fig.add_subplot(grid[3, 1])
    ax_resp = fig.add_subplot(grid[3, 2])
    ax_map = fig.add_subplot(grid[3, 3])
    metrics = compute_q_tcoll_metrics(result, data=data)

    x = result["x"]
    data_x_q = data_y_q = data_y_q_err = None
    data_x_t = data_y_t = data_y_t_err = None
    if data is not None:
        if "x_q" in data and "y_q" in data:
            data_x_q = np.asarray(data["x_q"], dtype=float)
            data_y_q = np.asarray(data["y_q"], dtype=float)
            data_y_q_err = data.get("y_q_err")
            ax_q.errorbar(data_x_q, data_y_q, yerr=data_y_q_err, fmt="o", ms=3, color="black", label="data Q")
        if "x_t" in data and "y_t" in data:
            data_x_t = np.asarray(data["x_t"], dtype=float)
            data_y_t = np.asarray(data["y_t"], dtype=float)
            data_y_t_err = data.get("y_t_err")
            ax_t.errorbar(data_x_t, data_y_t, yerr=data_y_t_err, fmt="o", ms=3, color="black", label="data tColl")
    ax_q.plot(x, result["charge_total"], color="tab:orange", label="Q total")
    ax_q.plot(x, result["charge_e"], color="darkblue", alpha=0.8, label="e")
    ax_q.plot(x, result["charge_h"], color="crimson", alpha=0.8, label="h")
    ax_q.set(ylabel="charge / NE", title="charge profile")
    ax_q.legend(frameon=False)

    if data_x_q is not None and data_y_q is not None:
        q_model_at_data = np.interp(data_x_q, x, result["charge_total"], left=np.nan, right=np.nan)
        ax_q_res.errorbar(data_x_q, data_y_q - q_model_at_data, yerr=data_y_q_err, fmt="o", ms=3, color="black")
    ax_q_res.axhline(0.0, color="black", lw=1)
    ax_q_res.set(xlabel="z focus / um", ylabel="data-model", title="charge residual")

    rc_label = f"total, RC tau={result['response']['rc_tau_ns']:.3g} ns"
    ax_t.plot(x, result["tcoll_rc"], color="black", lw=2.0, label=rc_label)
    ax_t.set(ylabel="duration / ns", title="pulse-duration profile")
    ax_t.legend(frameon=False)

    if data_x_t is not None and data_y_t is not None:
        t_model_at_data = np.interp(data_x_t, x, result["tcoll_rc"], left=np.nan, right=np.nan)
        ax_t_res.errorbar(data_x_t, data_y_t - t_model_at_data, yerr=data_y_t_err, fmt="o", ms=3, color="black")
    ax_t_res.axhline(0.0, color="black", lw=1)
    ax_t_res.set(xlabel="z focus / um", ylabel="data-model / ns", title="duration residual")

    response = result["response"]
    z = response["z"]
    ax_field.plot(z, response["efield"], color="tab:green")
    ax_field.set(xlabel="z / um", ylabel="field / (V/um)", title="electric field")
    ax_v.plot(z, response["vdrift_e"], color="darkblue", label="e")
    ax_v.plot(z, response["vdrift_h"], color="crimson", label="h")
    ax_v.set_yscale("log")
    ax_v.set(xlabel="z / um", ylabel="velocity / (um/ns)", title="drift velocity")
    ax_v.legend(frameon=False)
    ax_resp.plot(z, response["response_total"], color="black", label="sum")
    ax_resp.plot(z, response["response_e"], color="darkblue", label="e")
    ax_resp.plot(z, response["response_h"], color="crimson", label="h")
    ax_resp.set(xlabel="z / um", ylabel="response", title="material response")
    ax_resp.legend(frameon=False)
    ax_map.plot(z, response["pulse_duration_intrinsic"], color="black", label="total")
    ax_map.plot(z, response["pulse_duration_e_intrinsic"], color="darkblue", alpha=0.85, label="e")
    ax_map.plot(z, response["pulse_duration_h_intrinsic"], color="crimson", alpha=0.85, label="h")
    ax_map.set(xlabel="z / um", ylabel="duration / ns", title="intrinsic local duration")
    ax_map.legend(frameon=False)
    for ax in fig.axes:
        ax.grid(alpha=0.25)
    charge_metrics = metrics.get("charge") or {}
    tcoll_metrics = metrics.get("tcoll") or {}
    combined_metrics = metrics.get("combined") or {}
    include_t = bool(metrics.get("include_tcoll_in_cost", True))
    t_label = "included" if include_t else "ignored"
    fig.suptitle(
        "Combined cost: "
        f"NLL={_metric_text(combined_metrics.get('nll'))}, "
        f"chi2/N={_metric_text(combined_metrics.get('chi2_dof'))} "
        f"(tColl {t_label}); "
        f"Q chi2={_metric_text(charge_metrics.get('chi2'))}; "
        f"tColl chi2={_metric_text(tcoll_metrics.get('chi2'))}",
        fontsize=12,
    )
    return fig


def plot_model_waveforms(result, indices=None):
    import matplotlib.pyplot as plt

    waveforms = result.get("waveforms", {})
    if not waveforms:
        raise ValueError(
            "No stored waveforms in result. Set config['fit_options']['waveform_store_indices'] "
            "to a list of profile indices or 'auto' before simulate_q_tcoll_model()."
        )
    if indices is None:
        selected = sorted(waveforms)
    else:
        selected = [int(index) for index in indices if int(index) in waveforms]
    if not selected:
        raise ValueError("None of the requested waveform indices were stored")

    def with_zero_start(time, waveform):
        time = np.asarray(time, dtype=float)
        waveform = np.asarray(waveform, dtype=float)
        if time.size == 0:
            return np.asarray([0.0]), np.asarray([0.0])
        if time[0] == 0.0:
            return np.r_[0.0, time], np.r_[0.0, waveform]
        if time[0] > 0.0:
            return np.r_[0.0, time], np.r_[0.0, waveform]
        return time, waveform

    def normalize_waveform(waveform):
        waveform = np.asarray(waveform, dtype=float)
        peak = float(np.nanmax(np.abs(waveform))) if waveform.size else np.nan
        if not np.isfinite(peak) or peak <= 0.0:
            return waveform
        return waveform / peak

    def fall_time_to_fraction(time, waveform, fraction=0.05):
        time = np.asarray(time, dtype=float)
        waveform = np.asarray(waveform, dtype=float)
        if time.size < 3:
            return np.nan
        peak_index = int(np.nanargmax(waveform))
        peak = float(waveform[peak_index])
        if not np.isfinite(peak) or peak <= 0.0:
            return np.nan
        level = peak * float(fraction)
        right = waveform[peak_index:]
        cross = np.where(right <= level)[0]
        if not cross.size:
            return float(time[-1])
        j1 = peak_index + int(cross[0])
        j0 = max(peak_index, j1 - 1)
        y0, y1 = waveform[j0], waveform[j1]
        t0, t1 = time[j0], time[j1]
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
            return float(t1)
        return float(t0 + (level - y0) * (t1 - t0) / (y1 - y0))

    threshold = next(iter(waveforms.values())).get("threshold_percent", 5.0)
    x_limit_raw = 0.0
    x_limit_rc = 0.0
    for index in selected:
        wf = waveforms[index]
        t_raw, y_raw = with_zero_start(wf["time_ns"], wf["total"])
        t_rc, y_rc = with_zero_start(wf["time_ns"], wf["total_rc"])
        x_limit_raw = max(x_limit_raw, fall_time_to_fraction(t_raw, y_raw, threshold / 100.0))
        x_limit_rc = max(x_limit_rc, fall_time_to_fraction(t_rc, y_rc, threshold / 100.0))
    fallback_limit = max(float(np.nanmax(next(iter(waveforms.values()))["time_ns"])), 1.0)
    if not np.isfinite(x_limit_raw) or x_limit_raw <= 0.0:
        x_limit_raw = fallback_limit
    if not np.isfinite(x_limit_rc) or x_limit_rc <= 0.0:
        x_limit_rc = fallback_limit
    x_limit_raw *= 1.05
    x_limit_rc *= 1.05

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    ax_raw, ax_rc, ax_raw_norm, ax_rc_norm = axes.ravel()
    cmap = plt.get_cmap("viridis")
    z_values = np.asarray([waveforms[index]["x"] for index in selected], dtype=float)
    z_min_color = float(np.nanmin(z_values))
    z_max_color = float(np.nanmax(z_values))
    norm = plt.Normalize(z_min_color, z_max_color if z_max_color > z_min_color else z_min_color + 1.0)

    for index in selected:
        wf = waveforms[index]
        color = cmap(norm(float(wf["x"])))
        label = f"z={wf['x']:.2f} um"
        t_raw, y_raw = with_zero_start(wf["time_ns"], wf["total"])
        t_rc, y_rc = with_zero_start(wf["time_ns"], wf["total_rc"])
        ax_raw.plot(t_raw, y_raw, color=color, lw=1.0, alpha=0.9, label=label)
        ax_rc.plot(t_rc, y_rc, color=color, lw=1.0, alpha=0.9, label=label)
        ax_raw_norm.plot(t_raw, normalize_waveform(y_raw), color=color, lw=1.0, alpha=0.9)
        ax_rc_norm.plot(t_rc, normalize_waveform(y_rc), color=color, lw=1.0, alpha=0.9)

    ax_raw.set(title=f"total model WFs before RC, threshold={threshold:g}%", ylabel="induced current / arb.")
    ax_rc.set(title="total model WFs after RC", ylabel="filtered current / arb.")
    ax_raw_norm.set(title="normalized WFs before RC", xlabel="time / ns", ylabel="normalized amplitude")
    ax_rc_norm.set(title="normalized WFs after RC", xlabel="time / ns", ylabel="normalized amplitude")
    raw_margin = 0.04 * x_limit_raw
    rc_margin = 0.04 * x_limit_rc
    for ax in (ax_raw, ax_raw_norm):
        ax.set_xlim(-raw_margin, x_limit_raw)
    for ax in (ax_rc, ax_rc_norm):
        ax.set_xlim(-rc_margin, x_limit_rc)
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.015, fraction=0.03)
    cbar.set_label("focus z / um")
    return fig


def fit_trapping_q_tcoll_model(
    x_charge,
    y_charge,
    configuration,
    *,
    y_charge_sigma=None,
    x_tcoll=None,
    y_tcoll=None,
    y_tcoll_sigma=None,
    show_summary=True,
    show_fit=True,
    show_covariance=False,
    show_correlation=False,
    save_results=True,
    fit_name="q_tcoll_profile_roofit",
    output_dir="SiC_UVLED/Fits_RooFit/Q_TColl",
    do_useMINOS=False,
    params_MINOS=None,
    include_real=None,
    table_sigfigs=4,
    refit_fromLocalMinimumMINOS=False,
    show_roofit_output=False,
):
    """Fit charge and, optionally, pulse duration with the local RooFit/C++ NLL."""
    cfg = load_configuration(configuration)
    options = cfg.setdefault("fit_options", {})
    ignore_pulse = bool(options.get("ignore_pulse_duration_fit", False))
    ignore_wfs = bool(options.get("ignore_compute_WFs", False))
    include_tcoll = bool(options.get("include_tcoll_in_cost", True)) and not ignore_pulse

    if ignore_wfs:
        options["include_tcoll_in_cost"] = False
        include_tcoll = False

    roofit_cfg = copy.deepcopy(cfg)
    roofit_cfg["parameters"] = {
        name: copy.deepcopy(spec)
        for name, spec in cfg["parameters"].items()
        if name in MODEL_PARAMETER_NAMES
    }

    if not include_tcoll:
        output = fit_trapping_model(
            x_charge,
            y_charge,
            roofit_cfg,
            y_sigma=y_charge_sigma,
            show_summary=show_summary,
            show_fit=show_fit,
            show_covariance=show_covariance,
            show_correlation=show_correlation,
            save_results=save_results,
            fit_name=fit_name,
            output_dir=output_dir,
            do_useMINOS=do_useMINOS,
            params_MINOS=[] if params_MINOS is None else params_MINOS,
            include_real=include_real,
            table_sigfigs=table_sigfigs,
            refit_fromLocalMinimumMINOS=refit_fromLocalMinimumMINOS,
            show_roofit_output=show_roofit_output,
        )
        output["q_tcoll_fit_options"] = {
            "ignore_pulse_duration_fit": ignore_pulse,
            "ignore_compute_WFs": ignore_wfs,
            "include_tcoll_in_cost": False,
            "x_tcoll_provided": x_tcoll is not None,
            "y_tcoll_provided": y_tcoll is not None,
            "y_tcoll_sigma_provided": y_tcoll_sigma is not None,
            "fit_engine": "local RooFit/Minuit2 charge-profile C++ wrapper",
        }
        return output

    if x_tcoll is None or y_tcoll is None or y_tcoll_sigma is None:
        raise ValueError(
            "x_tcoll, y_tcoll and y_tcoll_sigma are required when "
            "fit_options['include_tcoll_in_cost']=True and "
            "fit_options['ignore_pulse_duration_fit']=False"
        )

    total_start = perf_counter()
    ROOT = _compile_cpp_model()
    config = load_fit_configuration(roofit_cfg)
    specs = config["parameters"]
    options = config["fit_options"]
    options["include_tcoll_in_cost"] = True
    options["ignore_pulse_duration_fit"] = False
    options["ignore_compute_WFs"] = False
    if do_useMINOS is not None:
        options["do_useMINOS"] = bool(do_useMINOS)
    if params_MINOS is not None:
        options["params_MINOS"] = params_MINOS
    if table_sigfigs is not None:
        options["table_sigfigs"] = int(table_sigfigs)
    if refit_fromLocalMinimumMINOS is not None:
        options["refit_fromLocalMinimumMINOS"] = bool(refit_fromLocalMinimumMINOS)
    if show_roofit_output is not None:
        options["show_roofit_output"] = bool(show_roofit_output)

    xq = np.asarray(x_charge, dtype=float)
    yq = np.asarray(y_charge, dtype=float)
    if xq.ndim != 1 or yq.shape != xq.shape:
        raise ValueError("x_charge and y_charge must be one-dimensional arrays with equal shape")
    sigma_q = _make_point_sigmas(yq, y_charge_sigma, options)

    xt = np.asarray(x_tcoll, dtype=float)
    yt = np.asarray(y_tcoll, dtype=float)
    sigma_t = np.asarray(y_tcoll_sigma, dtype=float)
    if xt.ndim != 1 or yt.shape != xt.shape or sigma_t.shape != xt.shape:
        raise ValueError("x_tcoll, y_tcoll and y_tcoll_sigma must be one-dimensional arrays with equal shape")
    if np.any(~np.isfinite(sigma_t)) or np.any(sigma_t <= 0.0):
        raise ValueError("y_tcoll_sigma must contain finite positive values")

    variables, parameter_list, fit_names, bounds, constraint_indices, constraint_means, constraint_sigmas = _make_roofit_parameters(ROOT, config)
    if not fit_names:
        raise ValueError("At least one effective parameter must have type='fit'")

    q_values = parameter_values(cfg)
    rc_tau = rc_tau_ns_from_values(q_values)
    waveform_dt = float(options.get("waveform_dt_ns", 0.02))
    threshold_percent = float(options.get("tColl_threshold_percent", 5.0))

    nll = ROOT.TrappingQTCollGaussianNLLV1(
        "trapping_q_tcoll_gaussian_nll",
        "trapping Q+tColl Gaussian NLL",
        parameter_list,
        _std_vector(xq),
        _std_vector(yq),
        _std_vector(sigma_q),
        _std_vector(xt),
        _std_vector(yt),
        _std_vector(sigma_t),
        ROOT.std.vector("int")(constraint_indices),
        _std_vector(constraint_means),
        _std_vector(constraint_sigmas),
        int(options.get("steps_per_active_region", 400)),
        int(options.get("n_z_grid", DEFAULT_N_Z_GRID)),
        bool(specs["EF_BiasVoltage"].get("enabled", False)),
        _field_model_kind(config),
        True,
        False,
        waveform_dt,
        threshold_percent,
        rc_tau,
    )

    minimizer = ROOT.RooMinimizer(nll)
    minimizer.setMinimizerType(str(options.get("minimizer_type", "Minuit2")))
    minimizer.setStrategy(int(options.get("strategy", 1)))
    minimizer.setPrintLevel(int(options.get("print_level", -1)))
    minimize_start = perf_counter()
    with _suppress_c_output(not bool(options.get("show_roofit_output", False))):
        status = int(minimizer.minimize(
            str(options.get("minimizer_type", "Minuit2")),
            str(options.get("minimizer_algorithm", "Minimize")),
        ))
    minimize_elapsed = perf_counter() - minimize_start
    hesse_errors = _errors_from_roofit_vars(variables, fit_names)

    minos_status = None
    minos_names = []
    minos_status_by_parameter = {}
    minos_raw_status_by_parameter = {}
    minos_elapsed_by_parameter = {}
    errors_minus = np.full(len(fit_names), np.nan, dtype=float)
    errors_plus = np.full(len(fit_names), np.nan, dtype=float)
    minos_elapsed = None
    if bool(options.get("do_useMINOS", False)):
        minos_names = _selected_minos_parameters(options.get("params_MINOS", []), fit_names)
        minos_start = perf_counter()
        fit_index = {name: index for index, name in enumerate(fit_names)}
        previous_raw_minos_status = int(status)
        best_fit_values = {name: float(variables[name].getVal()) for name in MODEL_PARAMETER_NAMES}
        for name in minos_names:
            for parameter_name, value in best_fit_values.items():
                variables[parameter_name].setVal(value)
            minos_parameters = ROOT.RooArgSet()
            minos_parameters.add(variables[name])
            parameter_start = perf_counter()
            with _suppress_c_output(not bool(options.get("show_roofit_output", False))):
                try:
                    raw_parameter_status = int(minimizer.minos(minos_parameters))
                except Exception:
                    raw_parameter_status = -1
            elapsed = perf_counter() - parameter_start
            if raw_parameter_status < 0:
                parameter_status = raw_parameter_status
            else:
                raw_delta = raw_parameter_status - previous_raw_minos_status
                parameter_status = raw_delta // 10 if raw_delta >= 0 and raw_delta % 10 == 0 else raw_parameter_status
            previous_raw_minos_status = raw_parameter_status if raw_parameter_status >= 0 else previous_raw_minos_status
            minos_raw_status_by_parameter[name] = raw_parameter_status
            minos_status_by_parameter[name] = parameter_status
            minos_elapsed_by_parameter[name] = elapsed
            if parameter_status == 0:
                index = fit_index[name]
                error_low = float(variables[name].getErrorLo())
                error_high = float(variables[name].getErrorHi())
                if np.isfinite(error_low) and error_low < 0.0:
                    errors_minus[index] = abs(error_low)
                if np.isfinite(error_high) and error_high > 0.0:
                    errors_plus[index] = error_high
        minos_elapsed = perf_counter() - minos_start
        minos_status = 0 if all(value == 0 for value in minos_status_by_parameter.values()) else 1

    fit_result = minimizer.save()
    parameters = _current_parameter_values(variables, config)
    parameters.update({name: q_values[name] for name in ("RC_capacitance_pF", "RC_resistance_ohm", "RC_extra_sigma_ns") if name in q_values})
    profile = ROOT.trapping_roofit_v5.evaluate_profile(
        _std_vector(xq),
        _std_vector([parameters[name] for name in MODEL_PARAMETER_NAMES]),
        int(options.get("steps_per_active_region", 400)),
        int(options.get("n_z_grid", DEFAULT_N_Z_GRID)),
        bool(specs["EF_BiasVoltage"].get("enabled", False)),
        _field_model_kind(config),
    )
    y_fit, y_fit_e, y_fit_h, y_fit_no_offset, y_fit_offset, response = _profile_result_to_dict(profile)
    t_fit = _to_numpy(ROOT.trapping_roofit_v5.evaluate_tcoll_profile(
        _std_vector(xt),
        _std_vector([parameters[name] for name in MODEL_PARAMETER_NAMES]),
        int(options.get("steps_per_active_region", 400)),
        int(options.get("n_z_grid", DEFAULT_N_Z_GRID)),
        bool(specs["EF_BiasVoltage"].get("enabled", False)),
        _field_model_kind(config),
        waveform_dt,
        threshold_percent,
        rc_tau,
        False,
    ))
    chi2_q = float(np.sum(((y_fit - yq) / sigma_q) ** 2))
    chi2_t = float(np.sum(((t_fit - yt) / sigma_t) ** 2))
    chi2 = chi2_q + chi2_t
    dof = max(yq.size + yt.size - len(fit_names), 1)
    covariance, correlation = _covariance_and_correlation(ROOT, fit_result, fit_names)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(fit_name)).strip("_") or "q_tcoll_roofit"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    generation_id = f"{timestamp}_{random.randint(1, 100)}_{safe_name}"
    output = {
        "result": fit_result,
        "minimize_status": status,
        "minos_status": minos_status,
        "minos_names": minos_names,
        "minos_status_by_parameter": minos_status_by_parameter,
        "minos_raw_status_by_parameter": minos_raw_status_by_parameter,
        "nll": float(nll.getVal()),
        "generation_id": generation_id,
        "fit_name": safe_name,
        "configuration": config,
        "initial_parameters": _parameter_values(specs),
        "real_parameters": _included_real_parameter_values(include_real),
        "parameters": parameters,
        "fit_names": fit_names,
        "fit_bounds": bounds,
        "covariance": covariance,
        "correlation": correlation,
        "errors": hesse_errors,
        "errors_minus": errors_minus,
        "errors_plus": errors_plus,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2 / dof,
        "chi2_charge": chi2_q,
        "chi2_tcoll": chi2_t,
        "x": xq,
        "y_data": yq,
        "y_sigma": sigma_q,
        "y_sigma_supplied": y_charge_sigma is not None,
        "y_fit": y_fit,
        "y_fit_e": y_fit_e,
        "y_fit_h": y_fit_h,
        "y_fit_no_offset": y_fit_no_offset,
        "y_fit_offset": y_fit_offset,
        "profile_offset": float(y_fit_offset),
        "x_tcoll": xt,
        "y_tcoll_data": yt,
        "y_tcoll_sigma": sigma_t,
        "y_tcoll_fit": t_fit,
        "material_response": response,
        "roofit_variables": variables,
        "roofit_nll": nll,
        "roofit_minimizer": minimizer,
        "refit_fromLocalMinimumMINOS": {"triggered": False, "parameter": None, "candidate": None},
        "timings": {
            "minimize_elapsed_s": minimize_elapsed,
            "minos_elapsed_s": minos_elapsed,
            "minos_elapsed_by_parameter_s": minos_elapsed_by_parameter,
            "total_elapsed_s": np.nan,
        },
    }
    output["q_tcoll_fit_options"] = {
        "ignore_pulse_duration_fit": ignore_pulse,
        "ignore_compute_WFs": ignore_wfs,
        "include_tcoll_in_cost": True,
        "x_tcoll_provided": True,
        "y_tcoll_provided": True,
        "y_tcoll_sigma_provided": True,
        "fit_engine": "local RooFit/Minuit2 Q+tColl C++ wrapper",
        "waveform_dt_ns": waveform_dt,
        "tColl_threshold_percent": threshold_percent,
        "rc_tau_ns": rc_tau,
    }
    output["timings"]["total_elapsed_s"] = perf_counter() - total_start
    if show_summary:
        _print_fit_summary(output)
        print(f"Q chi2/dof contribution = {chi2_q:.6g} / {max(yq.size - len(fit_names), 1)}")
        print(f"tColl chi2 contribution = {chi2_t:.6g} / {yt.size}")
    if save_results:
        _save_fit_diagnostic_plot(output, output_dir)
        _save_fit_result(output, output_dir)
    if show_fit:
        import matplotlib.pyplot as plt
        plot_fit_diagnostics(output)
        plt.show()
    if show_covariance or show_correlation:
        import matplotlib.pyplot as plt
        plot_covariance_or_correlation(output, use_correlation=show_correlation)
        plt.show()
    return output
