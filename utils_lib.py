def wu_rootfile(root_path: str, names: list, params: list, tree_name: str = "analysis"):
    """
    Crea o actualiza un ROOT file con un TTree 'analysis' de UNA entrada.

    - names: lista de nombres de ramas
    - params: lista de valores (escalares o vectores 1D)
    - Tipos se deducen desde params

    Caso especial (Variante B segura):
      - Si names/params está vacío, asegura que exista el tree con una rama dummy (_dummy).
      - Si ya existe, NO reescribe nada (idempotente).

    Soporta:
      Escalares: int, float, bool, str, numpy scalars
      Vectores 1D: np.ndarray, list, tuple
        - numéricos: float32/float64/int32/int64/bool
        - strings: list[str] o np.ndarray dtype string/object
    """
    import ROOT
    import numpy as np
    import os
    from array import array

    # ---------------------------
    # helpers
    # ---------------------------
    def is_vector_like(x):
        return isinstance(x, (list, tuple, np.ndarray))

    def to_numpy_1d(x):
        a = np.asarray(x)
        if a.ndim != 1:
            raise ValueError("Solo se soportan vectores 1D")
        return a

    def is_string_scalar(x):
        return isinstance(x, (str, np.str_))

    def is_string_vector(x):
        if isinstance(x, np.ndarray) and x.dtype.kind in ("U", "S", "O"):
            return True
        if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(v, str) for v in x):
            return True
        return False

    def make_scalar_buffer(pytype):
        # ROOT leaf types: I=int32, L=int64, F=float32, D=float64, O=bool
        if pytype in (bool, np.bool_):
            return array("b", [0]), "O"
        if pytype in (int, np.int32):
            return array("i", [0]), "I"
        if pytype in (np.int64,):
            return array("q", [0]), "L"
        if pytype in (float, np.float32):
            return array("f", [0.0]), "F"
        if pytype in (np.float64,):
            return array("d", [0.0]), "D"
        raise TypeError(f"Tipo escalar no soportado: {pytype}")

    def make_vector_numeric(dtype):
        dt = np.dtype(dtype)
        if dt == np.dtype("float64"):
            return ROOT.std.vector("double")()
        if dt == np.dtype("float32"):
            return ROOT.std.vector("float")()
        if dt == np.dtype("int64"):
            return ROOT.std.vector("long long")()
        if dt == np.dtype("int32"):
            return ROOT.std.vector("int")()
        if dt == np.dtype("bool"):
            return ROOT.std.vector("bool")()
        # fallback
        return ROOT.std.vector("double")()

    def fill_vector_numeric(vec, arr):
        vec.clear()
        for v in arr:
            vec.push_back(v.item() if hasattr(v, "item") else v)

    def read_old_entry(tree):
        """Lee la entrada 0 del tree viejo y devuelve dict name->value."""
        out = {}
        if not tree or tree.GetEntries() == 0:
            return out
        tree.GetEntry(0)
        for br in tree.GetListOfBranches():
            name = br.GetName()
            val = getattr(tree, name)
            cls = val.__class__.__name__ if hasattr(val, "__class__") else ""
            if "vector" in cls:
                out[name] = list(val)
            elif "string" in cls:
                out[name] = str(val)
            else:
                out[name] = val
        return out

    # ---------------------------
    # checks
    # ---------------------------
    if len(names) != len(params):
        raise ValueError("names y params deben tener la misma longitud")

    mode = "UPDATE" if os.path.exists(root_path) else "RECREATE"
    f = ROOT.TFile.Open(root_path, mode)
    if not f or f.IsZombie():
        raise RuntimeError(f"No se pudo abrir/crear: {root_path}")

    old_tree = f.Get(tree_name)

    # ============================================================
    # VARIANTE B (FIJA): si no hay parámetros, crea/asegura dummy
    # - Si ya existe el tree, no hagas nada (evita reescrituras)
    # - NO uses f.Delete(); solo kOverwrite cuando toque
    # ============================================================
    if len(names) == 0:
        if old_tree:  # ya está inicializado (o ya existe un analysis real)
            f.Close()
            return root_path

        f.cd()
        t = ROOT.TTree(tree_name, "Analysis base tree (single entry)")
        dummy = array("i", [1])
        t.Branch("_dummy", dummy, "_dummy/I")
        t.Fill()

        # OJO: sin Delete; overwrite es suficiente y evita segfaults
        t.Write("", ROOT.TObject.kOverwrite)
        f.Close()
        return root_path

    # ---------------------------
    # merge old + new (preserva lo viejo y sobreescribe lo nuevo)
    # ---------------------------
    old_vals = read_old_entry(old_tree)
    values = dict(old_vals)
    for n, p in zip(names, params):
        values[n] = p

    # Si existía el dummy y ahora metes ramas reales, lo quitamos
    # (al reconstruir el árbol, no lo incluimos salvo que lo pases tú)
    if "_dummy" in values and "_dummy" not in names:
        values.pop("_dummy", None)

    # ---------------------------
    # create new tree
    # ---------------------------
    f.cd()
    new_tree = ROOT.TTree(tree_name, "Analysis base tree (single entry)")

    buffers = {}
    kinds = {}

    for n, p in values.items():
        # strings escalares
        if is_string_scalar(p):
            s = ROOT.std.string(str(p))
            new_tree.Branch(n, s)
            buffers[n] = s
            kinds[n] = "scalar_string"
            continue

        # vector de strings
        if is_vector_like(p) and is_string_vector(p):
            v = ROOT.std.vector("string")()
            new_tree.Branch(n, v)
            buffers[n] = v
            kinds[n] = "vector_string"
            continue

        # vector numérico
        if is_vector_like(p):
            arr = to_numpy_1d(p)
            v = make_vector_numeric(arr.dtype)
            new_tree.Branch(n, v)
            buffers[n] = v
            kinds[n] = "vector_numeric"
            continue

        # escalar numérico/bool
        if isinstance(p, np.generic):
            p = p.item()
        buf, leaf = make_scalar_buffer(type(p))
        new_tree.Branch(n, buf, f"{n}/{leaf}")
        buffers[n] = buf
        kinds[n] = "scalar"

    # ---------------------------
    # fill entry
    # ---------------------------
    for n, p in values.items():
        k = kinds[n]
        if k == "scalar_string":
            buffers[n].assign(str(p))
        elif k == "vector_string":
            buffers[n].clear()
            if isinstance(p, np.ndarray):
                iterable = [str(x) for x in p.tolist()]
            else:
                iterable = [str(x) for x in p]
            for s in iterable:
                buffers[n].push_back(s)
        elif k == "vector_numeric":
            fill_vector_numeric(buffers[n], to_numpy_1d(p))
        else:
            if isinstance(p, np.generic):
                p = p.item()
            buffers[n][0] = p

    new_tree.Fill()

    # Sin Delete: overwrite evita el segfault de la secuencia Delete+Write
    new_tree.Write("", ROOT.TObject.kOverwrite)
    f.Close()

    return root_path


def wu_rootfileList(root_path: str, Gnames: list, Gparams: list, tree_name: str = "analysis"):
    for _path in root_path:
        wu_rootfile(_path, Gnames, Gparams, tree_name)

from pathlib import Path
import numpy as np

def fromDatafile_fill(
    file_names,
    root_dir,
    raw_dir,
    do_flipZ=True,
    do_invertSignal=False,
):
    """
    Unifica:
      - fill_datafile_meta_from_name: parsea metadatos desde el nombre
      - generate_data: lee datafile y genera z, x, y, LP, WFsRaw
    y lo guarda TODO en el ROOT file (TTree 'analysis') vía wu_rootfile().

    Parameters
    ----------
    file_names : list[str]
        Nombres base de los ficheros (sin extensión), ej:
        "20250606_1516_1MW2_400nm_450V_reps3_zscan_baseline_substrated"
    root_dir : str | Path
        Directorio donde están/irán los .root
    raw_dir : str | Path
        Directorio donde están los datafiles (sin extensión)
    do_flipZ : bool
        Si True: z = data[:,0] * -1000 ; si False: z = data[:,0] * 1000
    do_invertSignal : bool
        Si True: WFsRaw = -data[:,4:] ; si False: WFsRaw = data[:,4:]

    Returns
    -------
    dict
        Resumen por fichero: ok / error / paths
    """
    from utils_lib import wu_rootfile  # o importa donde tengas wu_rootfile

    root_dir = Path(root_dir)
    raw_dir = Path(raw_dir)

    summary = {}

    for file_name in file_names:
        try:
            name_lc = file_name.lower()
            tokens = name_lc.split("_")

            # --- metadatos desde el nombre ---
            timestamp = name_lc[:13] if len(name_lc) >= 13 else ""

            voltage_v = 0
            wavelength_nm = 0
            reps = 1
            scan_type = ""

            for t in tokens:
                if t.endswith("v") and t[:-1].isdigit():
                    voltage_v = int(t[:-1])

                if t.endswith("nm") and t[:-2].isdigit():
                    wavelength_nm = int(t[:-2])

                if t.startswith("reps") and t[4:].isdigit():
                    reps = int(t[4:])

                if not scan_type and ("scan" in t):
                    scan_type = t

            # --- paths ---
            root_path = root_dir / f"{file_name}.root"
            raw_path  = raw_dir  / file_name  # sin extensión

            # --- leer datafile (saltando 4 separadores de guiones) ---
            dash_count = 0
            data_lines = []

            with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if s and set(s) == {"-"}:
                        dash_count += 1
                        continue

                    if dash_count >= 4:
                        if s:
                            data_lines.append(s.replace(",", "."))

            if not data_lines:
                raise ValueError(f"No se encontraron datos en {raw_path}")

            data = np.loadtxt(data_lines, delimiter="\t")
            if data.ndim == 1:
                data = data.reshape(1, -1)

            # --- separar componentes (igual que tu generate_data) ---
            z = data[:, 0] * (-1000 if do_flipZ else 1000)
            x = data[:, 1] * 1000
            y = data[:, 2] * 1000
            LP = np.abs(data[:, 3])
            WFsRaw = (-1 * data[:, 4:]) if do_invertSignal else data[:, 4:]

            # --- preparar names/params para wu_rootfile ---
            # nombres = nombres exactos de variables / campos
            names = [
                "name",
                "timestamp",
                "voltage_v",
                "wavelength_nm",
                "reps",
                "scan_type",
                "dataPath",
                "rootPath",
                "do_flipZ",
                "do_invertSignal",
                "z",
                "x",
                "y",
                "LP",
                "WFsRaw",
            ]

            params = [
                name_lc,
                timestamp,
                voltage_v,
                wavelength_nm,
                reps,
                scan_type,
                str(raw_path),
                str(root_path),
                bool(do_flipZ),
                bool(do_invertSignal),
                z.astype(np.float64, copy=False),
                x.astype(np.float64, copy=False),
                y.astype(np.float64, copy=False),
                LP.astype(np.float64, copy=False),
                WFsRaw.astype(np.float64, copy=False).ravel(),  # 1D (importante)
            ]

            # ⚠️ WFsRaw originalmente es 2D (n_points, n_samples)
            # wu_rootfile soporta vectores 1D: lo guardamos "flatten".
            # Si quieres reconstruir luego, guardamos también shape:
            names += ["WFsRaw_shape0", "WFsRaw_shape1"]
            params += [int(WFsRaw.shape[0]), int(WFsRaw.shape[1])]

            # --- escribir ROOT ---
            wu_rootfile(str(root_path), names, params)

            summary[file_name] = {
                "ok": True,
                "root_path": str(root_path),
                "raw_path": str(raw_path),
                "n_rows": int(WFsRaw.shape[0]),
                "n_samples": int(WFsRaw.shape[1]),
            }

        except Exception as e:
            summary[file_name] = {
                "ok": False,
                "error": repr(e),
            }

    return summary
