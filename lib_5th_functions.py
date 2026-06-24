import glob
import os
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import ROOT
from plotly.subplots import make_subplots

from utils_lib import (
    fromDatafile_fill,
    getVals,
    reshape_paramReps,
    wu_rootfile,
    wu_rootfileList,
)
from analysis_lib import (
    analyze_signalsROI,
    analyze_wfsraw,
    correct_Signals,
    get_signalsROI,
    integrate_charge,
)


RAW_DATA = "SiC5thCamp/RawData/"
IMAGES = "SiC5thCamp//Images/"
ROOT_FILES_DIR = "SiC5thCamp/RootFiles/"


def get_5th_paths():
    return RAW_DATA, IMAGES, ROOT_FILES_DIR


def ensure_5th_directories():
    for directory in [RAW_DATA, IMAGES, ROOT_FILES_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")



def get_last_entries(path=None, text="line", search=None, n=10):
    if path is None:
        path = RAW_DATA

    pattern = f"{path}/*{text}*"
    entries = sorted(glob.glob(pattern))

    if search:
        entries = [e for e in entries if search in os.path.basename(e)]

    last = entries[-n:]

    names = [os.path.basename(e) for e in last]

    print(*names, sep="\n")
    return names

def generate_rootFiles(filenames, do_force=False, do_invert=False):
    root_files = [f"{ROOT_FILES_DIR}{file_name}.root" for file_name in filenames]
    root_files_full = root_files

    if not do_force:
        new_filenames = []
        new_root_files = []

        for i, root_file in enumerate(root_files):
            if not os.path.exists(root_file):
                new_filenames.append(filenames[i])
                new_root_files.append(root_file)

        filenames = new_filenames
        root_files = new_root_files

    print(*root_files, sep="\n")

    print("Generating root files...")

    print("DATAFILES:")
    for name in filenames:
        outpath = Path(ROOT_FILES_DIR) / f"{name}.root"
        wu_rootfile(
            str(outpath),
            ["dataPath"],
            [f"{RAW_DATA}{name}"],
            tree_name="Base",
        )

    print(filenames)
    print("DONE!")

    print("Loading initial parameters...")

    initial_pars = {
        "dt": 0.02,
        "aLPower": 1.00,
    }

    for key, value in initial_pars.items():
        wu_rootfileList(root_files, [key], [value], tree_name="Pars")

    fromDatafile_fill(filenames, ROOT_FILES_DIR, RAW_DATA, tree_name="Raw", do_invertSignal=do_invert)

    for root_path in root_files:
        vals = getVals(root_path, ["Raw:WFsRaw", "Pars:dt"])

        wfs_raw = vals["Raw:WFsRaw"]
        dt = float(vals["Pars:dt"])

        n_events = wfs_raw.shape[0]
        n_samples = wfs_raw.shape[1] if wfs_raw.ndim > 1 else 0

        events = np.arange(n_events, dtype=np.int32)
        t = np.arange(n_samples, dtype=np.int32) * dt

        wu_rootfile(
            root_path,
            ["n_events", "n_samples", "events", "t"],
            [n_events, n_samples, events, t],
            tree_name="Raw",
        )

        reshape_paramReps(root_path, "Raw:z")

    print("DONE!")

    return root_files_full


def _read_raw_waveforms(raw_path):
    dash_count = 0
    data_lines = []

    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and set(s) == {"-"}:
                dash_count += 1
                continue

            if dash_count >= 4 and s:
                data_lines.append(s.replace(",", "."))

    if not data_lines:
        raise ValueError(f"No se encontraron datos en {raw_path}")

    data = np.loadtxt(data_lines, delimiter="\t")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] <= 4:
        raise ValueError(f"No hay columnas de WFs en {raw_path}")

    return np.asarray(data[:, 4:], dtype=float)


def _raw_file_needs_inversion(file_name):
    raw_path = Path(RAW_DATA) / file_name
    wfs_raw = _read_raw_waveforms(raw_path)
    max_value = float(np.nanmax(wfs_raw))
    min_value = float(np.nanmin(wfs_raw))
    return max_value < abs(min_value)


def _auto_initial_wfPars_from_root(root_path, left_margin_ns=1.0, right_margin_ns=10.0):
    vals = getVals(root_path, ["Raw:WFsRaw", "Raw:t"])
    wfs_raw = np.asarray(vals["Raw:WFsRaw"], dtype=float)
    t = np.asarray(vals["Raw:t"], dtype=float)

    if wfs_raw.ndim == 1:
        wfs_raw = wfs_raw.reshape(1, -1)
    if wfs_raw.size == 0 or t.size == 0:
        raise ValueError(f"WFsRaw o t vacios en {root_path}")
    if wfs_raw.shape[1] != t.size:
        raise ValueError(
            f"Dimensiones incompatibles en {root_path}: "
            f"WFsRaw={wfs_raw.shape}, t={t.shape}"
        )

    max_per_wf = np.nanmax(wfs_raw, axis=1)
    if not np.any(np.isfinite(max_per_wf)):
        raise ValueError(f"No hay maximos finitos en {root_path}")

    wf = wfs_raw[int(np.nanargmax(max_per_wf))]
    peak_idx = int(np.nanargmax(wf))
    peak = float(wf[peak_idx])
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError(f"La WF de maxima amplitud no tiene pico positivo en {root_path}")

    threshold = 0.95 * peak

    def _crossing_left():
        left = wf[: peak_idx + 1]
        cross = np.where((left[:-1] <= threshold) & (left[1:] > threshold))[0]
        if cross.size == 0:
            return float(t[0])
        i = int(cross[-1])
        y0, y1 = wf[i], wf[i + 1]
        x0, x1 = t[i], t[i + 1]
        if y1 == y0:
            return float(x0)
        return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))

    def _crossing_right():
        right = wf[peak_idx:]
        cross = np.where((right[:-1] > threshold) & (right[1:] <= threshold))[0]
        if cross.size == 0:
            return float(t[-1])
        i = peak_idx + int(cross[0])
        y0, y1 = wf[i], wf[i + 1]
        x0, x1 = t[i], t[i + 1]
        if y1 == y0:
            return float(x1)
        return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))

    t_left = max(float(t[0]), _crossing_left() - float(left_margin_ns))
    t_right = min(float(t[-1]), _crossing_right() + float(right_margin_ns))

    if t_right <= t_left:
        raise ValueError(
            f"Ventana automatica invalida en {root_path}: "
            f"aTLeft={t_left}, aTRight={t_right}"
        )

    # aTBL marks the end of the baseline region used by analyze_wfsraw.
    a_tbl = t_left
    return [a_tbl, t_left, t_right]


def analyze_intensity(
    rf,
    charge_branch,
    xlim_left=None,
    xlim_right=None,
    rel_error=0.07,
):
    vals = getVals(rf, ["Raw:LP", "Raw:maxV", charge_branch])

    x = np.asarray(vals["Raw:LP"])
    maxV = np.asarray(vals["Raw:maxV"])
    charge = np.asarray(vals[charge_branch])

    lx = np.log(x)
    ly = np.log(charge)

    mask = np.ones(len(x), dtype=bool)

    if xlim_left is not None:
        mask &= x >= xlim_left

    if xlim_right is not None:
        mask &= x <= xlim_right

    lx_fit = lx[mask]
    ly_fit = ly[mask]

    if lx_fit.size == 0:
        raise ValueError(
            "No points selected for intensity fit. Check intensity_fit_limits."
        )

    print(f"Max amplitude  value for fit sel: {np.max(maxV[mask])} V")
    print(f"Max LaserPower value for fit sel: {np.max(x[mask])} V")

    sigma_y = rel_error * np.mean(ly_fit) / ly_fit

    p, cov = np.polyfit(
        lx_fit,
        ly_fit,
        1,
        w=1 / sigma_y,
        cov="unscaled",
    )

    b, a = p
    db = np.sqrt(cov[0, 0])

    ly_model = a + b * lx_fit

    chi2 = np.sum(((ly_fit - ly_model) / sigma_y) ** 2)
    ndof = len(ly_fit) - 2
    chi2_red = chi2 / ndof

    x_line = np.linspace(lx_fit.min(), lx_fit.max(), 200)
    y_line = a + b * x_line

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=lx, y=ly, mode="markers", name="Data"))

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Fit: b = {b:.3f} ± {db:.3f}, χ²/dof = {chi2_red:.2f}",
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title="log(Laser Power)",
        yaxis_title=f"log({charge_branch})",
    )

    fig.show()

    return a, b, db, chi2_red


def show_stdAnalysis(
    filenames,
    initial_wfPars: list,
    remove_roots=False,
    interp_nsamples=2,
    cor_mode="COR",
    sic_z_factor=2.83,
    plot_render="png",
    individual_x_limits=None,
    scan_type="zscan",
    intensity_charge_branch="Charge:Roi_masked",
    intensity_fit_limits=None,
    save_plots=False,
    write_html=False,
    invert_wfs=False,
    use_initial_motorPos=False,
    generate_globalPlots=True,
):
    """Generate ROOT files, run missing standard analyses, and show control plots."""
    filenames = list(filenames)
    root_files = [f"{ROOT_FILES_DIR}{file_name}.root" for file_name in filenames]
    charge_mode = cor_mode.upper()
    scan_type = str(scan_type).lower()
    if scan_type not in {"zscan", "iscan"}:
        raise ValueError("scan_type debe ser 'zscan' o 'iscan'.")
    if individual_x_limits is not None and len(individual_x_limits) != 2:
        raise ValueError(
            "individual_x_limits debe ser None o una lista/tupla [xmin, xmax]."
        )
    if intensity_fit_limits is not None and len(intensity_fit_limits) != 2:
        raise ValueError(
            "intensity_fit_limits debe ser None o una lista/tupla [xmin, xmax]."
        )

    def _has_param(root_path, spec):
        if ":" not in spec:
            return False
        tree_name, branch_name = spec.split(":", 1)
        try:
            f = ROOT.TFile.Open(root_path)
            if not f or f.IsZombie():
                return False
            tree = f.Get(tree_name)
            exists = bool(tree and tree.GetBranch(branch_name))
            f.Close()
            return exists
        except Exception:
            return False

    def _has_all_params(root_path, specs):
        return all(_has_param(root_path, spec) for spec in specs)

    def _safe_name(text):
        return Path(str(text)).stem.replace(" ", "_").replace("/", "_")

    def _timestamp_label(root_path):
        try:
            timestamp = str(getVals(root_path, ["Raw:timestamp"])["Raw:timestamp"])
        except Exception:
            timestamp = Path(root_path).stem
        parts = timestamp.split("_")
        if len(parts) >= 2 and len(parts[0]) >= 2:
            return f"{parts[0][-2:]}_{parts[1]}"
        return timestamp

    def _voltage_label(root_path):
        try:
            voltage = float(getVals(root_path, ["Raw:voltage_v"])["Raw:voltage_v"])
            return f"{voltage:g}V ({_timestamp_label(root_path)})"
        except Exception:
            return Path(root_path).stem

    def _plot_param_xy(
        root_files,
        x_param,
        y_param,
        labels,
        out_name,
        x_factor=1.0,
        y_index=None,
        grouped=False,
        style="line",
        show=True,
        x_limits=None,
        row_label_prefix=None,
        x_offset_param=None,
        x_offset_factor=1.0,
        normalize_y=False,
    ):
        mode_map = {
            "marker": "markers",
            "line": "lines",
            "markerline": "lines+markers",
        }
        mode = mode_map.get(style, "lines")
        fig = go.Figure()

        for root_path in root_files:
            params = [x_param, y_param]
            if x_offset_param is not None:
                params.append(x_offset_param)
            vals = getVals(root_path, params)
            x = np.asarray(vals[x_param], dtype=float) * float(x_factor)
            y = np.asarray(vals[y_param], dtype=float)
            if x_offset_param is not None:
                x = x + float(vals[x_offset_param]) * float(x_offset_factor)

            if y_index is not None:
                y = y[int(y_index)]
            if normalize_y:
                ymax = np.nanmax(y)
                if np.isfinite(ymax) and ymax != 0:
                    y = y / ymax

            trace_name = _voltage_label(root_path) if grouped else Path(root_path).stem

            if y.ndim == 1:
                fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=trace_name))
            elif y.ndim == 2:
                if x.ndim != 1 or y.shape[1] != x.shape[0]:
                    raise ValueError(
                        f"Dimensiones incompatibles para {y_param}: "
                        f"x={x.shape}, y={y.shape}"
                    )
                for idx, row in enumerate(y):
                    if row_label_prefix is None:
                        row_name = f"{trace_name}_evt{idx}"
                        row_showlegend = False
                    else:
                        row_name = f"{row_label_prefix}{idx + 1}"
                        row_showlegend = True
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=row,
                            mode=mode,
                            name=row_name,
                            showlegend=row_showlegend,
                        )
                    )
            else:
                raise ValueError(f"Solo se soportan arrays 1D o 2D para {y_param}")

        fig.update_layout(
            title=labels[0],
            xaxis_title=labels[1],
            yaxis_title=labels[2],
            width=800,
            height=600,
        )
        if x_limits is not None:
            fig.update_xaxes(range=list(x_limits))

        if save_plots:
            out_dir = Path(IMAGES)
            out_dir.mkdir(parents=True, exist_ok=True)
            png_path = out_dir / f"{out_name}.png"
            pio.write_image(fig, str(png_path))
            print(png_path)

        if write_html:
            out_dir = Path(IMAGES)
            out_dir.mkdir(parents=True, exist_ok=True)
            html_path = out_dir / f"{out_name}.html"
            fig.write_html(str(html_path))
            print(html_path)

        if show:
            if plot_render == "png":
                fig.show(renderer="png")
            else:
                fig.show()

        return fig

    print("Data files:")
    print(*filenames, sep="\n")

    if remove_roots:
        for root_path in root_files:
            if os.path.exists(root_path):
                os.remove(root_path)
                print(f"Removed {root_path}")

    root_files = generate_rootFiles(filenames, do_force=False, do_invert=invert_wfs)

    if use_initial_motorPos:
        missing_pos0 = [
            root_path for root_path in root_files if not _has_param(root_path, "Raw:pos0_1")
        ]
        if missing_pos0:
            missing_names = "\n".join(missing_pos0)
            raise ValueError(
                "use_initial_motorPos=True requiere la rama Raw:pos0_1. "
                "Regenera esos ROOT con remove_roots=True o vuelve a ejecutar "
                f"fromDatafile_fill para incluir la cabecera:\n{missing_names}"
            )

    wf_param_names = ["aTBL", "aTLeft", "aTRight"]
    wf_param_specs = [f"Raw:{name}" for name in wf_param_names]
    if len(initial_wfPars) != len(wf_param_names):
        raise ValueError(
            f"initial_wfPars debe tener {len(wf_param_names)} valores: "
            f"{wf_param_names}"
        )

    for root_path in root_files:
        if _has_all_params(root_path, wf_param_specs):
            print(f"{root_path}: initial WFs parameters already present, skip writing")
        else:
            wu_rootfile(root_path, wf_param_names, initial_wfPars, tree_name="Raw")
            print(f"{root_path}: initial WFs parameters written")

    for root_path in root_files:
        print(f"Analysis of {root_path} on progress ...")

        if _has_param(root_path, "Raw:TLeft"):
            print("  analyze_wfsraw: skip")
        else:
            analyze_wfsraw(root_path, tleft_frac=0.1)

        if _has_param(root_path, "Signal:signals"):
            print("  get_signalsROI: skip")
        else:
            get_signalsROI(root_path, interp_nsamples=interp_nsamples)

        if scan_type == "zscan":
            corrected_key = f"Signal:signals_{charge_mode}"
            if _has_param(root_path, corrected_key):
                print(f"  correct_Signals({charge_mode}): skip")
            else:
                correct_Signals(root_path, mode=charge_mode)

            if _has_param(root_path, "Signal:SNR"):
                print("  analyze_signalsROI: skip")
            else:
                analyze_signalsROI(root_path)

            charge_jobs = [
                ("Raw:t", "Raw:WFsRaw", "Charge:Raw"),
                ("Signal:t", "Signal:signals", "Charge:Roi"),
                ("Signal:t", "Signal:signals_masked", "Charge:Roi_masked"),
                ("Signal:t", f"Signal:signals_{charge_mode}", f"Charge:{charge_mode}"),
                (
                    "Signal:t",
                    f"Signal:signals_{charge_mode}_masked",
                    f"Charge:{charge_mode}_masked",
                ),
            ]
        else:
            charge_jobs = [("Signal:t", "Signal:signals_masked", intensity_charge_branch)]

        for t_spec, signal_spec, charge_spec in charge_jobs:
            if _has_param(root_path, charge_spec):
                if scan_type == "zscan":
                    print(f"  {charge_spec}: skip integrate + reshape")
                else:
                    print(f"  {charge_spec}: skip integrate")
                continue

            integrate_charge(
                root_path,
                t_spec,
                signal_spec,
                saveto=charge_spec,
                unitfactor=1,
            )
            if scan_type == "zscan":
                reshape_paramReps(root_path, charge_spec)
                print(f"  {charge_spec}: integrated + reshaped")
            else:
                print(f"  {charge_spec}: integrated")

        print("done!")

    if scan_type == "iscan":
        for root_path in root_files:
            print(f"Intensity analysis of {root_path}")
            if intensity_fit_limits is None:
                analyze_intensity(root_path, intensity_charge_branch)
            else:
                analyze_intensity(
                    root_path,
                    intensity_charge_branch,
                    xlim_left=float(np.exp(intensity_fit_limits[0])),
                    xlim_right=float(np.exp(intensity_fit_limits[1])),
                )
        print("DONE!")
        return root_files

    common_plots = [
        (
            "Raw:z_A",
            f"Charge:{charge_mode}_A",
            [f"Charge {charge_mode} averaged vs z", "z [um]", "Charge [a.u.]"],
            f"charge_{charge_mode}_A_vs_z_all",
            False,
        ),
        (
            "Raw:z_A",
            f"Charge:{charge_mode}_masked_A",
            [f"Charge {charge_mode} masked averaged vs z", "z [um]", "Charge [a.u.]"],
            f"charge_{charge_mode}_masked_A_vs_z_all",
            False,
        ),
        (
            "Raw:z_A",
            f"Charge:{charge_mode}_masked_A",
            [
                f"Charge {charge_mode} masked averaged vs z normalized",
                "z [um]",
                "Charge / max(Charge)",
            ],
            f"charge_{charge_mode}_masked_A_norm_vs_z_all",
            True,
        ),
    ]

    common_x_offset_param = "Raw:pos0_1" if use_initial_motorPos else None
    common_x_offset_factor = 1000.0 * sic_z_factor

    if generate_globalPlots:
        for x_param, y_param, labels, out_name, normalize_y in common_plots:
            print("SET PLOTS")
            _plot_param_xy(
                root_files,
                x_param=x_param,
                y_param=y_param,
                labels=labels,
                out_name=out_name,
                x_factor=sic_z_factor,
                grouped=True,
                style="markerline",
                x_offset_param=common_x_offset_param,
                x_offset_factor=common_x_offset_factor,
                normalize_y=normalize_y,
            )

    for root_path in root_files:
        stem = Path(root_path).stem
        individual_plots = [
            (
                "Signal:t",
                "Signal:signals",
                [stem, "t", "signal"],
                f"{_safe_name(stem)}_t_vs_signal",
                1.0,
                None,
                "line",
            ),
            (
                "Signal:t",
                "Signal:signals_masked",
                [stem, "t", "signal"],
                f"{_safe_name(stem)}_t_vs_signal_masked",
                1.0,
                None,
                "line",
            ),
            (
                "Raw:z",
                "Signal:TLeft",
                [stem, "z [um]", "TLeft [ns]"],
                f"{_safe_name(stem)}_TLeft_vs_z",
                sic_z_factor,
                None,
                "marker",
            ),
            (
                "Raw:z",
                "Signal:TRight",
                [stem, "z [um]", "TRight [ns]"],
                f"{_safe_name(stem)}_TRight_vs_z",
                sic_z_factor,
                None,
                "marker",
            ),
            (
                "Raw:z",
                "Signal:maxV",
                [stem, "z [um]", "maxV [V]"],
                f"{_safe_name(stem)}_maxV_vs_z",
                sic_z_factor,
                None,
                "markerline",
            ),
            (
                "Raw:z",
                "Signal:SNR",
                [stem, "z [um]", "SNR"],
                f"{_safe_name(stem)}_SNR_vs_z",
                sic_z_factor,
                None,
                "markerline",
            ),
            (
                "Raw:z",
                "Signal:tColl",
                [stem, "z [um]", "tColl[1] [ns]"],
                f"{_safe_name(stem)}_tColl1_vs_z",
                sic_z_factor,
                1,
                "marker",
            ),
        ]

        print("INDIVIDUAL PLOTS")

        for x_param, y_param, labels, out_name, x_factor, y_index, style in individual_plots:
            _plot_param_xy(
                [root_path],
                x_param=x_param,
                y_param=y_param,
                labels=labels,
                out_name=out_name,
                x_factor=x_factor,
                y_index=y_index,
                grouped=False,
                style=style,
                x_limits=individual_x_limits,
            )

        _plot_param_xy(
            [root_path],
            x_param="Raw:z_A",
            y_param=f"Charge:{charge_mode}_masked_R",
            labels=[stem, "z [um]", "Charge [a.u.]"],
            out_name=f"{_safe_name(stem)}_charge_{charge_mode}_masked_R_vs_z",
            x_factor=sic_z_factor,
            grouped=False,
            style="markerline",
            x_limits=individual_x_limits,
            row_label_prefix="rep",
        )

    print("DONE!")
    return root_files


def auto_Std_analysis(
    filenames,
    remove_roots=False,
    interp_nsamples=2,
    cor_mode="COR",
    sic_z_factor=2.83,
    plot_render="png",
    individual_x_limits=None,
    scan_type="zscan",
    intensity_charge_branch="Charge:Roi_masked",
    intensity_fit_limits=None,
    save_plots=False,
    write_html=False,
    use_initial_motorPos=False,
    generate_globalPlots=True,
    left_margin_ns=1.0,
    right_margin_ns=10.0,
):
    """
    Standard analysis with automatic signal polarity and ROI window selection.

    For new ROOT files, polarity is inferred from the raw data file before
    writing Raw:WFsRaw. If max(WFsRaw) < abs(min(WFsRaw)), the waveforms are
    inverted. The interpolation window is inferred from the waveform with the
    largest positive peak: crossings at 95% of the peak define the core window,
    then left_margin_ns and right_margin_ns are applied.

    Files that fail emit a warning and are skipped from the returned list.
    """
    successful_roots = []

    for file_name in list(filenames):
        root_path = f"{ROOT_FILES_DIR}{file_name}.root"

        try:
            if remove_roots and os.path.exists(root_path):
                os.remove(root_path)
                print(f"Removed {root_path}")

            if os.path.exists(root_path):
                print(f"{root_path}: existing ROOT, skip raw polarity detection")
            else:
                invert_wfs = _raw_file_needs_inversion(file_name)
                print(f"{file_name}: auto invert_wfs={invert_wfs}")
                generate_rootFiles([file_name], do_force=False, do_invert=invert_wfs)

            initial_wfPars = _auto_initial_wfPars_from_root(
                root_path,
                left_margin_ns=left_margin_ns,
                right_margin_ns=right_margin_ns,
            )
            wu_rootfile(
                root_path,
                ["aTBL", "aTLeft", "aTRight"],
                initial_wfPars,
                tree_name="Raw",
            )
            print(
                f"{root_path}: auto WFs parameters "
                f"aTBL={initial_wfPars[0]:.3f}, "
                f"aTLeft={initial_wfPars[1]:.3f}, "
                f"aTRight={initial_wfPars[2]:.3f}"
            )

            analyzed_roots = show_stdAnalysis(
                [file_name],
                initial_wfPars=initial_wfPars,
                remove_roots=False,
                interp_nsamples=interp_nsamples,
                cor_mode=cor_mode,
                sic_z_factor=sic_z_factor,
                plot_render=plot_render,
                individual_x_limits=individual_x_limits,
                scan_type=scan_type,
                intensity_charge_branch=intensity_charge_branch,
                intensity_fit_limits=intensity_fit_limits,
                save_plots=save_plots,
                write_html=write_html,
                invert_wfs=False,
                use_initial_motorPos=use_initial_motorPos,
                generate_globalPlots=generate_globalPlots,
            )
            successful_roots.extend(analyzed_roots)

        except Exception as exc:
            warnings.warn(
                f"auto_Std_analysis skipped {file_name}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

    return successful_roots


def normalize_rows(data):
    data = np.asarray(data, dtype=float)
    row_max = np.nanmax(data, axis=1, keepdims=True)
    row_max[~np.isfinite(row_max) | (row_max == 0)] = np.nan
    return data / row_max


def left_crossing_index(wf, fraction=0.95):
    wf = np.asarray(wf, dtype=float)
    if not np.any(np.isfinite(wf)):
        return None
    peak_idx = int(np.nanargmax(wf))
    peak = wf[peak_idx]
    if not np.isfinite(peak) or peak <= 0 or peak_idx == 0:
        return None

    threshold = float(fraction) * peak
    left = wf[: peak_idx + 1]
    cross = np.where((left[:-1] <= threshold) & (left[1:] > threshold))[0]
    if cross.size == 0:
        return peak_idx
    return int(cross[-1] + 1)


def left_crossing_index_95(wf):
    return left_crossing_index(wf, fraction=0.95)


def shift_with_nan(wf, shift):
    wf = np.asarray(wf, dtype=float)
    shifted = np.full_like(wf, np.nan, dtype=float)
    if shift == 0:
        return wf.copy()
    if shift > 0:
        shifted[shift:] = wf[:-shift]
    else:
        shifted[:shift] = wf[-shift:]
    return shifted


def align_at_left_fraction(data, target_index=20, fraction=0.95):
    aligned = []
    for wf in np.asarray(data, dtype=float):
        crossing_idx = left_crossing_index(wf, fraction=fraction)
        if crossing_idx is None:
            aligned.append(wf)
            continue
        aligned.append(shift_with_nan(wf, int(target_index) - crossing_idx))
    return np.asarray(aligned)


def align_at_left95(data, target_index=20):
    return align_at_left_fraction(data, target_index=target_index, fraction=0.95)


def average_repetitions(data, reps):
    data = np.asarray(data, dtype=float)
    reps = int(reps)
    if reps <= 1:
        return data

    n_events = data.shape[0]
    n_pos = n_events // reps
    if n_pos * reps != n_events:
        raise ValueError(f"Cannot reshape {n_events} events into {reps} reps")

    return np.nanmean(data.reshape(reps, n_pos, data.shape[1]), axis=0)


def z0_from_left_fraction(z, charge, fraction=0.5):
    z = np.asarray(z, dtype=float)
    charge = np.asarray(charge, dtype=float)
    max_idx = int(np.nanargmax(charge))
    peak = charge[max_idx]
    if not np.isfinite(peak) or peak == 0 or max_idx == 0:
        return float(z[max_idx])

    threshold = float(fraction) * peak
    left = charge[: max_idx + 1]
    cross = np.where((left[:-1] <= threshold) & (left[1:] > threshold))[0]
    if cross.size == 0:
        return float(z[0])

    i = int(cross[-1])
    y0, y1 = charge[i], charge[i + 1]
    x0, x1 = z[i], z[i + 1]
    if y1 == y0:
        return float(x0)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def z0_from_left50(z, charge):
    return z0_from_left_fraction(z, charge, fraction=0.5)


def blue_white_red_colorscale(white_level=0.5, color_levels=None):
    if color_levels is None:
        color_levels = [0.0, white_level, 1.0]

    if len(color_levels) != 3:
        raise ValueError("color_levels debe tener tres valores: [blue, white, red].")

    blue_level, white_level, red_level = [float(level) for level in color_levels]

    if not 0.0 <= blue_level < white_level < red_level <= 1.0:
        raise ValueError(
            "color_levels debe cumplir 0 <= blue < white < red <= 1."
        )

    colorscale = [[0.0, "blue"]]
    if blue_level > 0.0:
        colorscale.append([blue_level, "blue"])
    colorscale.append([white_level, "white"])
    if red_level < 1.0:
        colorscale.append([red_level, "red"])
    colorscale.append([1.0, "red"])

    return colorscale


def _resolve_colorscale(colorscale="Viridis", white_level=0.5, color_levels=None):
    if colorscale is None:
        return "Viridis"
    if isinstance(colorscale, str) and colorscale.lower() in {
        "blue_white_red",
        "bwr",
    }:
        return blue_white_red_colorscale(
            white_level=white_level,
            color_levels=color_levels,
        )
    return colorscale


def corrected_signal_heatmap_figure(
    root_path,
    colorscale="Viridis",
    white_level=0.5,
    color_levels=None,
    width=1200,
    height=800,
):
    colorscale = _resolve_colorscale(
        colorscale,
        white_level=white_level,
        color_levels=color_levels,
    )
    vals = getVals(root_path, ["Signal:signals_COR", "Signal:signals_COR_masked"])
    signals_cor = np.asarray(vals["Signal:signals_COR"], dtype=float)
    signals_cor_masked = np.asarray(vals["Signal:signals_COR_masked"], dtype=float)

    heatmaps = [
        (signals_cor, "Corrected signals"),
        (signals_cor_masked, "Corrected signals masked"),
        (normalize_rows(signals_cor), "Corrected signals normalized per WF"),
        (
            normalize_rows(signals_cor_masked),
            "Corrected masked normalized per WF",
        ),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[title for _, title in heatmaps],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, (data, _) in enumerate(heatmaps):
        row = idx // 2 + 1
        col = idx % 2 + 1
        fig.add_trace(
            go.Heatmap(
                z=data,
                x=np.arange(data.shape[1]),
                y=np.arange(data.shape[0]),
                colorscale=colorscale,
                colorbar=dict(title="a.u." if idx < 2 else "norm"),
                hovertemplate=(
                    "WF index: %{y}<br>"
                    "sample index: %{x}<br>"
                    "value: %{z}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Corrected signal heatmaps - {Path(root_path).stem}",
        width=width,
        height=height,
    )
    fig.update_xaxes(title_text="sample index")
    fig.update_yaxes(title_text="WF index")
    return fig


def aligned_reps_average_heatmap_figure(
    root_path,
    tleft_index_pos=20,
    sic_factor=2.83,
    colorscale="Viridis",
    white_level=0.5,
    color_levels=None,
    width=1100,
    height=1500,
    title_prefix=None,
):
    colorscale = _resolve_colorscale(
        colorscale,
        white_level=white_level,
        color_levels=color_levels,
    )
    vals = getVals(
        root_path,
        [
            "Signal:t",
            "Signal:signals_COR",
            "Signal:signals_COR_masked",
            "Raw:reps",
            "Raw:z_A",
            "Charge:COR_masked_A",
            "Raw:voltage_v",
        ],
    )

    signal_t = np.asarray(vals["Signal:t"], dtype=float)
    signals_cor = np.asarray(vals["Signal:signals_COR"], dtype=float)
    signals_cor_masked = np.asarray(vals["Signal:signals_COR_masked"], dtype=float)
    reps = int(vals["Raw:reps"])
    z_air = np.asarray(vals["Raw:z_A"], dtype=float)
    charge_profile = np.asarray(vals["Charge:COR_masked_A"], dtype=float)
    voltage = float(vals["Raw:voltage_v"])

    signals_cor_aligned = align_at_left95(
        signals_cor,
        target_index=tleft_index_pos,
    )
    signals_cor_masked_aligned = align_at_left95(
        signals_cor_masked,
        target_index=tleft_index_pos,
    )
    signals_cor_norm_aligned = align_at_left95(
        normalize_rows(signals_cor),
        target_index=tleft_index_pos,
    )
    signals_cor_masked_norm_aligned = align_at_left95(
        normalize_rows(signals_cor_masked),
        target_index=tleft_index_pos,
    )

    signals_cor_avg = average_repetitions(signals_cor_aligned, reps)
    signals_cor_masked_avg = average_repetitions(signals_cor_masked_aligned, reps)
    signals_cor_norm_avg = average_repetitions(signals_cor_norm_aligned, reps)
    signals_cor_masked_norm_avg = average_repetitions(
        signals_cor_masked_norm_aligned,
        reps,
    )

    z0_air = z0_from_left50(z_air, charge_profile)
    z_sic_corr = (z_air - z0_air) * float(sic_factor)

    heatmaps = [
        (signals_cor_avg, "Aligned corrected signals, reps averaged"),
        (signals_cor_masked_avg, "Aligned corrected masked signals, reps averaged"),
        (
            signals_cor_norm_avg,
            "Aligned corrected signals normalized per WF, reps averaged",
        ),
        (
            signals_cor_masked_norm_avg,
            "Aligned corrected masked normalized per WF, reps averaged",
        ),
    ]

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=[title for _, title in heatmaps],
        vertical_spacing=0.06,
    )

    for idx, (data, _) in enumerate(heatmaps):
        fig.add_trace(
            go.Heatmap(
                z=data.T,
                x=z_sic_corr,
                y=signal_t,
                colorscale=colorscale,
                colorbar=dict(title="a.u." if idx < 2 else "norm"),
                hovertemplate=(
                    "z(SiC)-z0: %{x:.2f} um<br>"
                    "t: %{y:.3f} ns<br>"
                    "value: %{z}<extra></extra>"
                ),
            ),
            row=idx + 1,
            col=1,
        )

    if title_prefix is None:
        title_prefix = Path(root_path).stem

    fig.update_layout(
        title=f"{title_prefix} {voltage:g} V - aligned and reps-averaged heatmaps",
        width=width,
        height=height,
    )
    fig.update_xaxes(title_text="z - z0 in SiC [um]")
    fig.update_yaxes(title_text="t [ns]")
    return fig


def show_corrected_signal_heatmaps(root_path, **kwargs):
    fig = corrected_signal_heatmap_figure(root_path, **kwargs)
    fig.show()
    return fig


def show_aligned_reps_average_heatmaps(root_path, **kwargs):
    fig = aligned_reps_average_heatmap_figure(root_path, **kwargs)
    fig.show()
    return fig
