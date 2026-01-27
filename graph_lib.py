from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from utils_lib import getVals


from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as pc
from utils_lib import getVals


def plot_root_param_xy(
    out_dir,
    out_name,
    root_files,
    x_param,
    y_param,
    labels,
    show=True,
    style="line",
    limits=None,
    write_html=True,
    render="png"
):
    """
    Genera un PNG y un HTML interactivo a partir de parámetros en ROOT.

    Parameters
    ----------
    out_dir : str | Path
        Directorio de salida.
    out_name : str
        Nombre base (sin extensión) de los outputs.
    root_files : list[str]
        Lista de ROOT files a procesar.
    x_param : str
        Especificador "tree:param" para el eje X.
    y_param : str
        Especificador "tree:param" para el eje Y.
    labels : list[str]
        [titulo, xlabel, ylabel].
    show : bool
        Si True, muestra la figura como PNG (no interactivo).
    style : str
        "marker", "line" o "markerline".
    limits : list[float] | None
        [xmin, xmax, ymin, ymax]. Si min==max==0 para un eje,
        se ignora el set del límite de ese eje.
    write_html : bool
        Si True, genera también el HTML interactivo.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_map = {
        "marker": "markers",
        "line": "lines",
        "markerline": "lines+markers",
    }
    mode = mode_map.get(style, "lines")

    fig = go.Figure()

    for root_path in root_files:
        vals = getVals(root_path, [x_param, y_param])
        x = np.asarray(vals[x_param])
        y = np.asarray(vals[y_param])

        if y.ndim == 1:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode=mode,
                    name=Path(root_path).stem,
                )
            )
            continue

        if y.ndim == 2:
            if x.ndim != 1 or y.shape[1] != x.shape[0]:
                raise ValueError(
                    "Dimensiones incompatibles: y.shape[1] debe coincidir con x.shape[0]."
                )
            for idx, row in enumerate(y):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=row,
                        mode=mode,
                        name=f"{Path(root_path).stem}_evt{idx}",
                        showlegend=False,
                    )
                )
            continue

        raise ValueError("Solo se soportan y de 1D o 2D para plot_root_param_xy")

    title = labels[0] if labels and len(labels) > 0 else ""
    xlabel = labels[1] if labels and len(labels) > 1 else x_param
    ylabel = labels[2] if labels and len(labels) > 2 else y_param

    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)

    if limits:
        xmin, xmax, ymin, ymax = limits
        if not (xmin == 0 and xmax == 0):
            fig.update_xaxes(range=[xmin, xmax])
        if not (ymin == 0 and ymax == 0):
            fig.update_yaxes(range=[ymin, ymax])

    png_path = out_dir / f"{out_name}.png"
    pio.write_image(fig, str(png_path))
    print(png_path)
    
    html_path = None
    if write_html:
        html_path = out_dir / f"{out_name}.html"
        fig.write_html(str(html_path))
        print(html_path)

    if show:
        if render == "png":
            fig.show(renderer="png")
        else:
            fig.show()

    return str(png_path), str(html_path) if html_path else None


def plot_WFs(
    root_path,
    signals: str,
    legend_pars: list,
    subsection: list = None,
    cut_limits: list = None,
    norm_signals: str = "",
    colormap_val: str = "",
):
    """
    Plot waveform signals from a ROOT file using plotly.
    """
    if subsection is None:
        subsection = []
    if cut_limits is None:
        cut_limits = []

    specs = [signals]
    specs += legend_pars
    if norm_signals:
        specs.append(norm_signals)

    cut_specs = []
    for entry in cut_limits:
        if entry:
            cut_specs.append(entry.split(";", 1)[0])
    specs += cut_specs

    color_scale = ""
    color_param = ""
    if colormap_val:
        parts = colormap_val.split(";", 1)
        color_scale = parts[0].strip()
        if len(parts) > 1:
            color_param = parts[1].strip()
            if color_param:
                specs.append(color_param)

    vals = getVals(root_path, specs)
    signals_mat = np.asarray(vals[signals])
    n_events = signals_mat.shape[0]
    n_samples = signals_mat.shape[1]

    if norm_signals:
        norm_vals = np.asarray(vals[norm_signals])
        if norm_vals.ndim == 0:
            norm_vals = np.full(n_events, float(norm_vals))
        signals_mat = signals_mat / norm_vals[:, None]

    indices = list(range(n_events))
    if subsection:
        indices = []
        for spec in subsection:
            parts = spec.split(":")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else n_events
            step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
            indices.extend(list(range(start, end, step)))
        indices = [i for i in indices if 0 <= i < n_events]

    if cut_limits:
        mask = np.ones(n_events, dtype=bool)
        for entry in cut_limits:
            if not entry:
                continue
            spec, vmin, vmax = (entry.split(";") + ["", ""])[:3]
            param_vals = np.asarray(vals[spec])
            if vmin and vmin.lower() != "none":
                mask &= param_vals >= float(vmin)
            if vmax and vmax.lower() != "none":
                mask &= param_vals <= float(vmax)
        indices = [i for i in indices if mask[i]]

    colors = None
    if color_scale:
        if color_param:
            color_vals = np.asarray(vals[color_param])
        else:
            color_vals = np.arange(n_events)
        cmin = np.nanmin(color_vals)
        cmax = np.nanmax(color_vals) if np.nanmax(color_vals) != cmin else cmin + 1
        colors = [
            pc.sample_colorscale(
                color_scale, float((color_vals[i] - cmin) / (cmax - cmin))
            )[0]
            for i in indices
        ]

    fig = go.Figure()
    x = np.arange(n_samples)

    for j, idx in enumerate(indices):
        legend_parts = []
        for spec in legend_pars:
            val = vals[spec]
            if np.asarray(val).ndim == 0:
                legend_parts.append(f"{spec.split(':', 1)[1]}: {float(val)}")
            else:
                legend_parts.append(f"{spec.split(':', 1)[1]}: {val[idx]}")
        legend = " | ".join(legend_parts) if legend_parts else f"evt {idx}"

        line_color = colors[j] if colors else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=signals_mat[idx],
                mode="lines",
                name=legend,
                line={"color": line_color} if line_color else None,
                showlegend=True,
            )
        )

    fig.update_layout(xaxis_title="sample", yaxis_title=signals)
    fig.show()