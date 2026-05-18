from pathlib import Path
import numpy as np


def wu_rootfile(root_path: str, names: list, params: list, tree_name: str):
    """
    Versión DataFrame-like de escritura ROOT usando ROOT.RDF.FromNumpy + Snapshot.

    Mantiene la misma API que utils_lib.wu_rootfile:
    - mergea entrada previa (single-entry) + valores nuevos
    - sobrescribe ramas existentes con los valores nuevos
    - mantiene una única entrada en el TTree

    Nota: ROOT.RDF.FromNumpy soporta bien columnas numéricas. Para ramas string
    y algunos tipos no soportados por FromNumpy, se usa un fallback interno
    a escritura TTree clásica para preservar compatibilidad funcional.
    """
    import ROOT
    import os
    from array import array

    def parse_scalar(x):
        if isinstance(x, np.generic):
            return x.item()
        return x

    def is_vector_like(x):
        return isinstance(x, (list, tuple, np.ndarray))

    def is_string_scalar(x):
        return isinstance(x, (str, np.str_))

    def is_string_vector(x):
        if isinstance(x, np.ndarray) and x.dtype.kind in ("U", "S", "O"):
            return True
        if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(v, str) for v in x):
            return True
        return False

    def vector_to_list(obj):
        cls = obj.__class__.__name__ if hasattr(obj, "__class__") else ""
        if "vector" in cls:
            return [vector_to_list(v) for v in obj]
        return obj

    def read_old_entry(path, tname):
        out = {}
        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie():
            return out
        t = f.Get(tname)
        if not t or t.GetEntries() == 0:
            f.Close()
            return out
        t.GetEntry(0)
        for br in t.GetListOfBranches():
            n = br.GetName()
            v = getattr(t, n)
            cls = v.__class__.__name__ if hasattr(v, "__class__") else ""
            if "vector" in cls:
                out[n] = np.asarray(vector_to_list(v))
            elif "string" in cls:
                out[n] = str(v)
            else:
                out[n] = v
        f.Close()
        return out

    def try_rdf_write(path, tname, values):
        cols = {}
        for n, p in values.items():
            if is_string_scalar(p) or is_string_vector(p):
                return False

            if is_vector_like(p):
                arr = np.asarray(p)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                cols[n] = np.array([arr], dtype=object)
            else:
                s = parse_scalar(p)
                if isinstance(s, bool):
                    cols[n] = np.array([s], dtype=np.bool_)
                elif isinstance(s, int):
                    cols[n] = np.array([s], dtype=np.int64)
                elif isinstance(s, float):
                    cols[n] = np.array([s], dtype=np.float64)
                else:
                    return False

        try:
            # ROOT espera columnas con longitud consistente (aquí siempre 1 fila)
            df = ROOT.RDF.FromNumpy(cols)
            opts = ROOT.RDF.RSnapshotOptions()
            opts.fMode = "RECREATE"
            df.Snapshot(tname, path, list(cols.keys()), opts)
            return True
        except Exception:
            return False

    def fallback_classic_write(path, tname, values):
        f = ROOT.TFile.Open(path, "RECREATE")
        if not f or f.IsZombie():
            raise RuntimeError(f"No se pudo crear: {path}")

        t = ROOT.TTree(tname, "Analysis base tree (single entry)")

        buffers = {}
        kinds = {}

        for n, p in values.items():
            if is_string_scalar(p):
                s = ROOT.std.string(str(p))
                t.Branch(n, s)
                buffers[n] = s
                kinds[n] = "scalar_string"
                continue

            if is_vector_like(p) and is_string_vector(p):
                v = ROOT.std.vector("string")()
                t.Branch(n, v)
                buffers[n] = v
                kinds[n] = "vector_string"
                continue

            if is_vector_like(p):
                arr = np.asarray(p)
                if arr.ndim == 1:
                    dt = arr.dtype
                    if dt == np.dtype("float64"):
                        cpp = "double"
                    elif dt == np.dtype("float32"):
                        cpp = "float"
                    elif dt == np.dtype("int32"):
                        cpp = "int"
                    elif dt == np.dtype("int64"):
                        cpp = "long long"
                    elif dt == np.dtype("bool"):
                        cpp = "bool"
                    else:
                        cpp = "double"
                        arr = arr.astype(np.float64)
                    v = ROOT.std.vector(cpp)()
                    t.Branch(n, v)
                    buffers[n] = v
                    kinds[n] = ("vector_numeric", arr.dtype)
                else:
                    # Fallback conservador para arrays N-D: guardar como vector<double> flatten + shape
                    flat = arr.astype(np.float64, copy=False).ravel()
                    v = ROOT.std.vector("double")()
                    t.Branch(n, v)
                    buffers[n] = v
                    kinds[n] = ("vector_flat_nd", tuple(arr.shape))

                    shp = ROOT.std.vector("int")()
                    t.Branch(f"{n}__shape", shp)
                    buffers[f"{n}__shape"] = shp
                    kinds[f"{n}__shape"] = ("shape", tuple(arr.shape))
                continue

            s = parse_scalar(p)
            if isinstance(s, bool):
                buf = array("b", [0]); leaf = "O"
            elif isinstance(s, int):
                buf = array("q", [0]); leaf = "L"
            elif isinstance(s, float):
                buf = array("d", [0.0]); leaf = "D"
            else:
                raise TypeError(f"Tipo no soportado en fallback: {n} -> {type(s)}")
            t.Branch(n, buf, f"{n}/{leaf}")
            buffers[n] = buf
            kinds[n] = "scalar"

        for n, p in values.items():
            k = kinds[n]
            if k == "scalar_string":
                buffers[n].assign(str(p))
            elif k == "vector_string":
                buffers[n].clear()
                seq = p.tolist() if isinstance(p, np.ndarray) else p
                for s in seq:
                    buffers[n].push_back(str(s))
            elif isinstance(k, tuple) and k[0] == "vector_numeric":
                arr = np.asarray(p, dtype=k[1])
                buffers[n].clear()
                for v in arr:
                    vv = v.item() if hasattr(v, "item") else v
                    buffers[n].push_back(vv)
            elif isinstance(k, tuple) and k[0] == "vector_flat_nd":
                arr = np.asarray(p, dtype=np.float64).ravel()
                buffers[n].clear()
                for v in arr:
                    vv = v.item() if hasattr(v, "item") else v
                    buffers[n].push_back(vv)
                shp = buffers[f"{n}__shape"]
                shp.clear()
                for d in k[1]:
                    shp.push_back(int(d))
            elif k == "scalar":
                s = parse_scalar(p)
                buffers[n][0] = s

        t.Fill()
        t.Write("", ROOT.TObject.kOverwrite)
        f.Close()

    if len(names) != len(params):
        raise ValueError("names y params deben tener la misma longitud")
    if not tree_name:
        raise ValueError("tree_name es obligatorio")

    root_path = str(root_path)

    if len(names) == 0:
        # comportamiento equivalente: crear tree base con dummy solo si no existe
        if os.path.exists(root_path):
            old = read_old_entry(root_path, tree_name)
            if old:
                return root_path
        fallback_classic_write(root_path, tree_name, {"_dummy": 1})
        return root_path

    old_vals = read_old_entry(root_path, tree_name) if os.path.exists(root_path) else {}
    values = dict(old_vals)
    for n, p in zip(names, params):
        values[n] = p

    if "_dummy" in values and "_dummy" not in names:
        values.pop("_dummy", None)

    ok = try_rdf_write(root_path, tree_name, values)
    if not ok:
        fallback_classic_write(root_path, tree_name, values)

    return root_path


def wu_rootfileList(root_path: list, Gnames: list, Gparams: list, tree_name: str):
    for _path in root_path:
        wu_rootfile(_path, Gnames, Gparams, tree_name)


def fromDatafile_fill(
    file_names,
    root_dir,
    raw_dir,
    tree_name: str,
    do_flipZ=True,
    do_invertSignal=False,
    raise_on_error=False,
):
    """
    Misma API que utils_lib.fromDatafile_fill, escribiendo con wu_rootfile (DF backend).
    """
    root_dir = Path(root_dir)
    raw_dir = Path(raw_dir)

    summary = {}

    for file_name in file_names:
        try:
            name_lc = file_name.lower()
            tokens = name_lc.split("_")

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

            root_path = root_dir / f"{file_name}.root"
            raw_path = raw_dir / file_name

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

            z = data[:, 0] * (-1000 if do_flipZ else 1000)
            x = data[:, 1] * 1000
            y = data[:, 2] * 1000
            LP = np.abs(data[:, 3])
            WFsRaw = (-1 * data[:, 4:]) if do_invertSignal else data[:, 4:]

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
                WFsRaw.astype(np.float64, copy=False),
            ]

            wu_rootfile(str(root_path), names, params, tree_name)

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
            print(f"[ERROR] {file_name}: {e}")
            if raise_on_error:
                raise

    return summary


def getVals(root_path: str, keys: list):
    """
    Lee parámetros de un ROOT file usando especificadores "tree:param".

    Mantiene misma API de utils_lib.getVals. Intenta ruta DataFrame (`RDataFrame.AsNumpy`)
    y usa fallback clásico si aplica.
    """
    import ROOT

    def parse_key(spec: str):
        for sep in (":", "/", "."):
            if sep in spec:
                tree, param = spec.split(sep, 1)
                tree = tree.strip()
                param = param.strip()
                if tree and param:
                    return tree, param
        raise ValueError(f"Formato inválido para '{spec}'. Usa 'tree:param'.")

    def vector_to_list(obj):
        cls = obj.__class__.__name__ if hasattr(obj, "__class__") else ""
        if "vector" in cls:
            return [vector_to_list(v) for v in obj]
        return obj

    out = {}

    grouped = {}
    for spec in keys:
        t, p = parse_key(spec)
        grouped.setdefault(t, []).append((spec, p))

    for tree_name, reqs in grouped.items():
        cols = [p for _, p in reqs]

        used_df = False
        try:
            rdf = ROOT.RDataFrame(tree_name, root_path)
            data = rdf.AsNumpy(cols)
            for spec, p in reqs:
                arr = data[p]
                if len(arr) == 0:
                    raise ValueError(f"El TTree '{tree_name}' no tiene entradas")
                v = arr[0]
                if isinstance(v, (np.ndarray, list, tuple)):
                    out[spec] = np.asarray(v)
                else:
                    out[spec] = v.item() if isinstance(v, np.generic) else v
            used_df = True
        except Exception:
            used_df = False

        if used_df:
            continue

        f = ROOT.TFile.Open(root_path)
        if not f or f.IsZombie():
            raise RuntimeError(f"No se pudo abrir: {root_path}")
        t = f.Get(tree_name)
        if not t:
            f.Close()
            raise KeyError(f"No existe el TTree '{tree_name}' en {root_path}")
        if t.GetEntries() == 0:
            f.Close()
            raise ValueError(f"El TTree '{tree_name}' no tiene entradas")

        t.GetEntry(0)
        for spec, p in reqs:
            if not hasattr(t, p):
                f.Close()
                raise KeyError(f"No existe la rama '{p}' en '{tree_name}'")
            val = getattr(t, p)
            cls = val.__class__.__name__ if hasattr(val, "__class__") else ""
            if "vector" in cls:
                arr = np.asarray(vector_to_list(val))
                shape_name = f"{p}__shape"
                if hasattr(t, shape_name):
                    shp = np.asarray(vector_to_list(getattr(t, shape_name)), dtype=int)
                    if shp.size > 1:
                        try:
                            arr = arr.reshape(tuple(shp.tolist()))
                        except Exception:
                            pass
                out[spec] = arr
            elif "string" in cls:
                out[spec] = str(val)
            else:
                out[spec] = val

        f.Close()

    return out
