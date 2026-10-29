import os
import shutil
import pandas as pd
from .paths import data_path

DATOS_DIR = data_path("Datos")
CLASIFICADOS_DIR = data_path("Historico")

CLASIFICADOS_EXCEL = os.path.join(DATOS_DIR, "Historico.xlsx")
SEGUROS_EXCEL = os.path.join(DATOS_DIR, "Seguros.xlsx")
DUDOSOS_EXCEL = os.path.join(DATOS_DIR, "Dudosos.xlsx")


def actualizar_clasificados(clasificados_excel=None, seguros_excel=None,
                            dudosos_excel=None, clasificados_dir=None):
    """Sincroniza Historico.xlsx / Seguros.xlsx / Dudosos.xlsx con lo que
    realmente haya en Historico/.

    Los cuatro parámetros permiten apuntar a otro juego de rutas que las
    constantes del módulo; se usan desde ActualizarThread, que resuelve las
    rutas con runtime_paths (parcheables en tests), y quedan en None para el
    uso normal.
    """
    clas_excel = clasificados_excel or CLASIFICADOS_EXCEL
    seg_excel  = seguros_excel or SEGUROS_EXCEL
    dud_excel  = dudosos_excel or DUDOSOS_EXCEL
    clas_dir   = clasificados_dir or CLASIFICADOS_DIR

    print("🔄 Sincronizando con carpetas...")

    if not os.path.exists(clas_excel):
        print("❌ No existe Historico.xlsx")
        return

    df_clas = pd.read_excel(clas_excel)
    nombre_col = df_clas.columns[0]

    df_seg = pd.read_excel(seg_excel) if os.path.exists(seg_excel) else pd.DataFrame()
    df_dud = pd.read_excel(dud_excel) if os.path.exists(dud_excel) else pd.DataFrame()

    # ---- Escanear estructura de carpetas ----
    if not os.path.isdir(clas_dir):
        print("⚠ Carpeta Historico/ no existe aún. No hay nada que sincronizar.")
        return

    # Collect every physical location for each filename.
    # all_locs: fname -> list of (clase_dir, is_seguro, full_path)
    all_locs: dict = {}

    for clase_dir in os.listdir(clas_dir):
        clase_path = os.path.join(clas_dir, clase_dir)
        if not os.path.isdir(clase_path) or clase_dir == "_DUDOSOS":
            continue
        for fname in os.listdir(clase_path):
            fp = os.path.join(clase_path, fname)
            if os.path.isfile(fp):
                all_locs.setdefault(fname, []).append((clase_dir, True, fp))

    dudosos_path = os.path.join(clas_dir, "_DUDOSOS")
    if os.path.isdir(dudosos_path):
        for clase_dir in os.listdir(dudosos_path):
            clase_path = os.path.join(dudosos_path, clase_dir)
            if not os.path.isdir(clase_path):
                continue
            for fname in os.listdir(clase_path):
                fp = os.path.join(clase_path, fname)
                if os.path.isfile(fp):
                    all_locs.setdefault(fname, []).append((clase_dir, False, fp))

    # Resolve duplicates: keep the newest copy, delete the older ones.
    duplicates_removed = 0
    folder_state = {}  # fname -> (clase, is_seguro)
    for fname, locs in all_locs.items():
        if len(locs) == 1:
            clase_dir, is_seguro, _ = locs[0]
        else:
            locs.sort(key=lambda t: os.path.getmtime(t[2]), reverse=True)
            clase_dir, is_seguro, _ = locs[0]
            for _, _, old_path in locs[1:]:
                os.remove(old_path)
                duplicates_removed += 1
                print(f"🗑 Duplicado eliminado: {old_path}")
        folder_state[fname] = (clase_dir, is_seguro)

    # ---- Dudoso reclasificado → promover a seguro ----
    # Si un dudoso está en una clase distinta a la del excel,
    # el usuario lo reclasificó → mover a carpeta segura
    promovidos = 0
    if not df_dud.empty:
        for _, row in df_dud.iterrows():
            fname = str(row[nombre_col])
            if fname not in folder_state:
                continue
            clase_carpeta, is_seguro = folder_state[fname]
            if is_seguro:
                continue
            clase_excel = str(row["class_pred"])
            if clase_carpeta != clase_excel:
                src = os.path.join(clas_dir, "_DUDOSOS", clase_carpeta, fname)
                dst_dir = os.path.join(clas_dir, clase_carpeta)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src, os.path.join(dst_dir, fname))
                folder_state[fname] = (clase_carpeta, True)
                promovidos += 1

    # ---- Recalcular conjuntos después de promociones ----
    archivos_en_carpetas = set(folder_state.keys())
    seguros_en_carpetas = {f for f, (_, s) in folder_state.items() if s}
    dudosos_en_carpetas = {f for f, (_, s) in folder_state.items() if not s}

    # ---- Historico.xlsx: retag + eliminar ausentes ----
    antes_clas = len(df_clas)
    df_clas = df_clas[df_clas[nombre_col].astype(str).isin(archivos_en_carpetas)]
    eliminados_clas = antes_clas - len(df_clas)

    cambios_clas = 0
    for i, row in df_clas.iterrows():
        fname = str(row[nombre_col])
        clase, _ = folder_state[fname]
        if row["class_pred"] != clase:
            df_clas.at[i, "class_pred"] = clase
            df_clas.at[i, "confidence"] = 1.0
            cambios_clas += 1

    # ---- Seguros.xlsx: retag + eliminar ausentes + agregar nuevos ----
    if not df_seg.empty:
        antes_seg = len(df_seg)
        df_seg = df_seg[df_seg[nombre_col].astype(str).isin(seguros_en_carpetas)]
        eliminados_seg = antes_seg - len(df_seg)
    else:
        eliminados_seg = 0

    retagged_seg = 0
    seg_existentes = set()
    if not df_seg.empty:
        for i, row in df_seg.iterrows():
            fname = str(row[nombre_col])
            seg_existentes.add(fname)
            clase, _ = folder_state[fname]
            if row["class_pred"] != clase:
                df_seg.at[i, "class_pred"] = clase
                df_seg.at[i, "confidence"] = 1.0
                retagged_seg += 1

    # Índice por nombre para no filtrar df_clas entero por cada archivo nuevo
    # (los dos loops de abajo hacían un scan O(n) por archivo).
    df_clas_by_name = df_clas.drop_duplicates(subset=nombre_col, keep="first").copy()
    df_clas_by_name[nombre_col] = df_clas_by_name[nombre_col].astype(str)
    df_clas_by_name = df_clas_by_name.set_index(nombre_col, drop=False)

    # Se juntan las filas nuevas en una lista y se concatena una sola vez al
    # final: concatenar adentro del loop recopia el DataFrame acumulado en
    # cada vuelta (O(n²) en la cantidad de archivos nuevos).
    nuevos_seg = seguros_en_carpetas - seg_existentes
    filas_seg = []
    for fname in nuevos_seg:
        clase, _ = folder_state[fname]
        if fname not in df_clas_by_name.index:
            continue
        fila = df_clas_by_name.loc[fname].copy()
        fila["class_pred"] = clase
        fila["confidence"] = 1.0
        filas_seg.append(fila)
    if filas_seg:
        df_seg = pd.concat([df_seg, pd.DataFrame(filas_seg)], ignore_index=True)

    # ---- Dudosos.xlsx: eliminar los que ya no están en Dudosos/ + agregar nuevos ----
    if not df_dud.empty:
        antes_dud = len(df_dud)
        df_dud = df_dud[df_dud[nombre_col].astype(str).isin(dudosos_en_carpetas)]
        eliminados_dud = antes_dud - len(df_dud)
    else:
        eliminados_dud = 0

    dud_existentes = set(df_dud[nombre_col].astype(str)) if not df_dud.empty else set()
    nuevos_dud = dudosos_en_carpetas - dud_existentes
    filas_dud = []
    for fname in nuevos_dud:
        clase, _ = folder_state[fname]
        if fname not in df_clas_by_name.index:
            continue
        fila = df_clas_by_name.loc[fname].copy()
        fila["class_pred"] = clase
        filas_dud.append(fila)
    if filas_dud:
        df_dud = pd.concat([df_dud, pd.DataFrame(filas_dud)], ignore_index=True)

    # ---- Guardar ----
    df_clas.reset_index(drop=True).to_excel(clas_excel, index=False)
    df_seg.reset_index(drop=True).to_excel(seg_excel, index=False)
    df_dud.reset_index(drop=True).to_excel(dud_excel, index=False)

    print(f"✔ Duplicados de archivo eliminados: {duplicates_removed}")
    print(f"✔ Retagged en Historico: {cambios_clas}")
    print(f"✔ Eliminados de Historico: {eliminados_clas}")
    print(f"✔ Dudosos promovidos a seguros: {promovidos}")
    print(f"✔ Nuevos en Seguros: {len(nuevos_seg)}")
    print(f"✔ Retagged en Seguros: {retagged_seg}")
    print(f"✔ Eliminados de Seguros: {eliminados_seg}")
    print(f"✔ Nuevos en Dudosos: {len(nuevos_dud)}")
    print(f"✔ Eliminados de Dudosos: {eliminados_dud}")
