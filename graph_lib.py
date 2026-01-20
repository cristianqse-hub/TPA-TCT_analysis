from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from utils_lib import getVals


from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

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

    html_path = None
    if write_html:
        html_path = out_dir / f"{out_name}.html"
        fig.write_html(str(html_path))

    if show:
        fig.show(renderer="png")

    return str(png_path), str(html_path) if html_path else None
