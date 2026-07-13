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
    return config


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

    if getattr(ROOT, "TrappingGaussianNLLV4", None) is not None:
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

namespace trapping_roofit_v4 {

enum ParameterIndex {
    BM_z0 = 0, BM_zRight, BM_zR0, BM_z_Aberr,
    BM_CoefA, BM_CoefB, BM_area, BM_scaleAmp, BM_scaleOffset,
    MV_beta_e, MV_vsat_e, MV_mu0_e,
    MV_beta_h, MV_vsat_h, MV_mu0_h,
    EF_BiasVoltage, EF_CoefA, EF_CoefB, EF_CoefC, EF_z0,
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
    bool voltage_enabled
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
    if (voltage_enabled) {
        const double center = p[EF_z0];
        const double u_left = -center;
        const double u_right = width - center;
        const double quadratic_integral = p[EF_CoefA] * (std::pow(u_right, 3) - std::pow(u_left, 3)) / 3.0;
        const double linear_integral = p[EF_CoefB] * (u_right * u_right - u_left * u_left) / 2.0;
        field_c = (p[EF_BiasVoltage] - quadratic_integral - linear_integral) / width;
    }

    out.efield.resize(steps);
    out.mobility_e.resize(steps);
    out.mobility_h.resize(steps);
    out.vdrift_mue.resize(steps);
    out.vdrift_muh.resize(steps);
    const double field_center_abs = z0_shifted + p[EF_z0];
    const double field_floor = 1e-2;
    for (int i = 0; i < steps; ++i) {
        const double u = out.z[i] - field_center_abs;
        const double efield = p[EF_CoefA] * u * u + p[EF_CoefB] * u + field_c;
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

} // namespace trapping_roofit_v4

class TrappingGaussianNLLV4 : public RooAbsReal {
public:
    TrappingGaussianNLLV4() {}

    TrappingGaussianNLLV4(
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
        bool voltage_enabled
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
          voltage_enabled_(voltage_enabled)
    {
        parameters_.add(parameters);
    }

    TrappingGaussianNLLV4(const TrappingGaussianNLLV4& other, const char* name = nullptr)
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
          voltage_enabled_(other.voltage_enabled_)
    {}

    TObject* clone(const char* newname) const override { return new TrappingGaussianNLLV4(*this, newname); }

protected:
    double evaluate() const override
    {
        std::vector<double> p(trapping_roofit_v4::N_PARAMETERS);
        for (int i = 0; i < trapping_roofit_v4::N_PARAMETERS; ++i) {
            const auto* value = dynamic_cast<const RooAbsReal*>(parameters_.at(i));
            if (!value) return 1e100;
            p[i] = value->getVal();
        }

        const auto profile = trapping_roofit_v4::evaluate_profile(
            x_, p, steps_per_active_region_, n_z_grid_, voltage_enabled_
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
            if (index < 0 || index >= trapping_roofit_v4::N_PARAMETERS) continue;
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
};
        '''
    )
    return ROOT


def _parameter_values(parameter_specs, fit_names=None, fit_vector=None):
    return _base._parameter_values(parameter_specs, fit_names, fit_vector)


def _effective_fit_names(parameter_specs):
    disabled = {"EF_CoefC"}
    return [name for name in _base._effective_fit_names(parameter_specs) if name not in disabled]


def _resolved_fit_bounds(parameter_specs, fit_names, fit_options):
    return _base._resolved_fit_bounds(parameter_specs, fit_names, fit_options)


def _configuration_values(config):
    values = _parameter_values(config["parameters"])
    values["EF_CoefC"] = _derived_field_constant(values)
    return np.array([values[name] for name in MODEL_PARAMETER_NAMES], dtype=float)


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
    result = ROOT.trapping_roofit_v4.evaluate_profile(
        _std_vector(x_vec),
        _std_vector(_configuration_values(config)),
        int(options.get("steps_per_active_region", 400)),
        int(options.get("n_z_grid", DEFAULT_N_Z_GRID)),
        bool(config["parameters"]["EF_BiasVoltage"].get("enabled", False)),
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


def _current_parameter_values(variables):
    values = {}
    for name in MODEL_PARAMETER_NAMES:
        values[name] = float(variables[name].getVal())
    values["EF_CoefC"] = _derived_field_constant(values)
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
    axis_velocity = figure.add_subplot(grid[0, 2])
    axis_response = figure.add_subplot(grid[1, 2])
    axis_field = figure.add_subplot(grid[2, 2])

    profile_offset = float(output.get("profile_offset", output.get("y_fit_offset", 0.0)))
    y_data_plot = np.asarray(output["y_data"], dtype=float) - profile_offset
    y_fit_plot = np.asarray(output["y_fit"], dtype=float) - profile_offset
    y_fit_e_plot = np.asarray(output["y_fit_e"], dtype=float)
    y_fit_h_plot = np.asarray(output["y_fit_h"], dtype=float)

    axis_profile.errorbar(
        x, y_data_plot, yerr=output["y_sigma"], fmt="o", ms=3,
        color="black", ecolor="0.6", elinewidth=0.8, capsize=0, label="data",
    )
    axis_profile.plot(x, y_fit_plot, color="tab:orange", lw=2,
                      label=f"fit, chi2/dof = {output['chi2_dof']:.3f}")
    if y_fit_e_plot.shape == x.shape:
        axis_profile.plot(x, y_fit_e_plot, color="darkblue", lw=1.5, alpha=0.8, label="e")
    if y_fit_h_plot.shape == x.shape:
        axis_profile.plot(x, y_fit_h_plot, color="crimson", lw=1.5, alpha=0.8, label="h")
    axis_profile.set(xlabel="focus position z / um", ylabel="charge / NE", title=f"RooFit ID: {output['generation_id']}")
    axis_profile.legend(frameon=False)

    residual = y_data_plot - y_fit_plot
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

    if response_v_e.shape == response_z.shape:
        axis_velocity.plot(response_z, response_v_e, color="darkblue", label="e")
    if response_v_h.shape == response_z.shape:
        axis_velocity.plot(response_z, response_v_h, color="crimson", label="h")
    axis_velocity.set(title="effective drift velocity", xlabel="z / um", ylabel="velocity / (um/ns)")
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
            "model_library": "trapping_model_RooFit_li.py",
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

    nll = ROOT.TrappingGaussianNLLV4(
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
                candidate_values = _current_parameter_values(variables)
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

    parameters = _current_parameter_values(variables)
    parameters["_EF_BiasVoltage_enabled"] = True

    profile = ROOT.trapping_roofit_v4.evaluate_profile(
        _std_vector(x),
        _std_vector([parameters[name] for name in MODEL_PARAMETER_NAMES]),
        int(options.get("steps_per_active_region", 400)),
        int(options.get("n_z_grid", DEFAULT_N_Z_GRID)),
        bool(specs["EF_BiasVoltage"].get("enabled", False)),
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
