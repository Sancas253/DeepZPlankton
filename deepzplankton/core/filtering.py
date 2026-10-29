"""Split Historico/ en seguros/dudosos según la confianza del modelo."""
import os
import shutil

import pandas as pd

from .excel_io import upsert_excel_by_key
from .paths import data_path

DATOS_DIR = data_path("Datos")
CLASIFICADOS_DIR = data_path("Historico")

CLASIFICADOS_EXCEL = os.path.join(DATOS_DIR, "Historico.xlsx")
SEGUROS_EXCEL = os.path.join(DATOS_DIR, "Seguros.xlsx")
DUDOSOS_EXCEL = os.path.join(DATOS_DIR, "Dudosos.xlsx")


def filtrar_dudosos(UMBRAL=0.95, clasificados_excel=None, clasificados_dir=None):

    if not (0.0 <= UMBRAL <= 1.0):
        raise ValueError(f"UMBRAL debe estar entre 0 y 1, recibido: {UMBRAL}")

    clas_excel = clasificados_excel or CLASIFICADOS_EXCEL
    clas_dir   = clasificados_dir   or CLASIFICADOS_DIR

    print("🔍 Leyendo Excel de clasificaciones...")

    if not os.path.exists(clas_excel):
        print(f"❌ No se encontró {os.path.basename(clas_excel)}")
        return

    df = pd.read_excel(clas_excel)

    # asegurar tipos correctos
    df["segment"] = df["segment"].astype(str)
    df["class_pred"] = df["class_pred"].astype(str)

    if "confidence" not in df.columns:
        print("❌ Falta la columna confidence")
        return

    # ---------------- FILTRAR SOLO NUEVOS ----------------
    existing_segments = set()
    if os.path.exists(SEGUROS_EXCEL):
        existing_segments.update(
            pd.read_excel(SEGUROS_EXCEL, usecols=["segment"])["segment"].astype(str)
        )
    if os.path.exists(DUDOSOS_EXCEL):
        existing_segments.update(
            pd.read_excel(DUDOSOS_EXCEL, usecols=["segment"])["segment"].astype(str)
        )

    df_new = df[~df["segment"].isin(existing_segments)]

    df_seguros = df_new[df_new["confidence"] >= UMBRAL].reset_index(drop=True)
    df_dudosos = df_new[df_new["confidence"] < UMBRAL].reset_index(drop=True)

    upsert_excel_by_key(SEGUROS_EXCEL, df_seguros)
    upsert_excel_by_key(DUDOSOS_EXCEL, df_dudosos)

    print(f"✅ Seguros nuevos: {len(df_seguros)}")
    print(f"✅ Dudosos nuevos: {len(df_dudosos)}")

    # ---------------- MAPEO (solo nuevos) ----------------
    seg_a_clase = dict(zip(df_new["segment"], df_new["class_pred"]))

    seguros_set = set(df_seguros["segment"])
    dudosos_set = set(df_dudosos["segment"])

    print("\n📂 Reorganizando imágenes...")

    if not os.path.isdir(clas_dir):
        print(f"⚠ Carpeta {os.path.basename(clas_dir)}/ no existe aún. No hay imágenes que mover.")
        return

    movidos_dudosos = 0

    for clase_dir in os.listdir(clas_dir):

        clase_path = os.path.join(clas_dir, clase_dir)

        if not os.path.isdir(clase_path) or clase_dir == "_DUDOSOS":
            continue

        for fname in os.listdir(clase_path):

            origen = os.path.join(clase_path, fname)

            if not os.path.isfile(origen):
                continue

            if fname not in seg_a_clase:
                continue

            clase = seg_a_clase[fname]

            if fname in dudosos_set:

                destino = os.path.join(clas_dir, "_DUDOSOS", clase)
                os.makedirs(destino, exist_ok=True)

                shutil.move(origen, os.path.join(destino, fname))
                movidos_dudosos += 1

    print(f"\n✅ Seguros (en carpeta): {len(seguros_set)}")
    print(f"✅ Movidos a Dudosos: {movidos_dudosos}")


def refilter_all(UMBRAL=0.95):
    """Re-apply UMBRAL to every entry in Historico.xlsx.

    Unlike filtrar_dudosos (which only processes new entries), this function
    moves images that changed state under the new threshold in both directions:
    seguros → dudosos and dudosos → seguros.
    """
    if not (0.0 <= UMBRAL <= 1.0):
        raise ValueError(f"UMBRAL debe estar entre 0 y 1, recibido: {UMBRAL}")

    print(f"🔄 Reaplicando umbral {UMBRAL} a toda la base...")

    if not os.path.exists(CLASIFICADOS_EXCEL):
        print("❌ No se encontró Historico.xlsx")
        return

    df = pd.read_excel(CLASIFICADOS_EXCEL)
    df["segment"] = df["segment"].astype(str)
    df["class_pred"] = df["class_pred"].astype(str)

    if "confidence" not in df.columns:
        print("❌ Falta la columna confidence")
        return

    df_seguros = df[df["confidence"] >= UMBRAL].reset_index(drop=True)
    df_dudosos = df[df["confidence"] < UMBRAL].reset_index(drop=True)

    df_seguros.to_excel(SEGUROS_EXCEL, index=False)
    df_dudosos.to_excel(DUDOSOS_EXCEL, index=False)

    seguros_set = set(df_seguros["segment"])
    dudosos_set = set(df_dudosos["segment"])
    seg_a_clase = dict(zip(df["segment"], df["class_pred"]))

    if not os.path.isdir(CLASIFICADOS_DIR):
        print("⚠ Carpeta Historico/ no existe aún.")
        return

    movidos_a_seguros = 0
    movidos_a_dudosos = 0

    # Promote images in _DUDOSOS that now exceed the threshold
    dudosos_dir = os.path.join(CLASIFICADOS_DIR, "_DUDOSOS")
    if os.path.isdir(dudosos_dir):
        for clase_dir in os.listdir(dudosos_dir):
            clase_path = os.path.join(dudosos_dir, clase_dir)
            if not os.path.isdir(clase_path):
                continue
            for fname in list(os.listdir(clase_path)):
                origen = os.path.join(clase_path, fname)
                if not os.path.isfile(origen) or fname not in seguros_set:
                    continue
                clase = seg_a_clase.get(fname, clase_dir)
                destino_dir = os.path.join(CLASIFICADOS_DIR, clase)
                os.makedirs(destino_dir, exist_ok=True)
                shutil.move(origen, os.path.join(destino_dir, fname))
                movidos_a_seguros += 1

    # Demote images in class folders that now fall below the threshold
    for clase_dir in os.listdir(CLASIFICADOS_DIR):
        clase_path = os.path.join(CLASIFICADOS_DIR, clase_dir)
        if not os.path.isdir(clase_path) or clase_dir == "_DUDOSOS":
            continue
        for fname in list(os.listdir(clase_path)):
            origen = os.path.join(clase_path, fname)
            if not os.path.isfile(origen) or fname not in dudosos_set:
                continue
            clase = seg_a_clase.get(fname, clase_dir)
            destino_dir = os.path.join(CLASIFICADOS_DIR, "_DUDOSOS", clase)
            os.makedirs(destino_dir, exist_ok=True)
            shutil.move(origen, os.path.join(destino_dir, fname))
            movidos_a_dudosos += 1

    print(f"✅ Seguros totales: {len(df_seguros)}")
    print(f"✅ Dudosos totales: {len(df_dudosos)}")
    print(f"↑ Promovidos a seguros: {movidos_a_seguros}")
    print(f"↓ Movidos a dudosos: {movidos_a_dudosos}")
