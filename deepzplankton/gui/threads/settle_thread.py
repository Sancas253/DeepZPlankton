import os
import shutil

from PyQt5.QtCore import QThread, pyqtSignal

from .. import runtime_paths as rt


class ActualizarThread(QThread):
    """El botón "Actualizar" — unifica lo que antes eran dos botones.

    1. Asienta lo que haya en EnRevision/ hacia Historico/ (el commit final
       de una tanda revisada).
    2. Reconcilia Historico/ con sus Excel (lo que antes era "Actualizar" a
       secas).

    Cada parte se saltea sola si no tiene nada que hacer: si EnRevision/ está
    vacío sólo se reconcilia, y la reconciliación es un no-op barato cuando
    nadie tocó Historico/ a mano.
    """

    finished = pyqtSignal()
    log      = pyqtSignal(str)

    def __init__(self, umbral):
        super().__init__()
        self.umbral = umbral

    def run(self):
        try:
            import pandas as pd

            from ...core.classifier import sin_confianza
            from ...core.excel_io import normalize_prediction_columns
            from ...core.files import enlazar_o_copiar
            from ...core.filtering import filtrar_dudosos
            from ...core.update import actualizar_clasificados

            self.log.emit("=== ACTUALIZAR ===")

            # El Excel se lee primero porque su columna de nombres es la que
            # manda para deshacer el prefijo de confianza que llevan los
            # archivos de EnRevision/ (ver core.naming.sin_confianza).
            df_proc = None
            nombres_reales = set()
            if os.path.exists(rt.procesado_excel):
                df_proc = normalize_prediction_columns(pd.read_excel(rt.procesado_excel))
                nombres_reales = set(df_proc.iloc[:, 0].astype(str))

            # Recolectar TODOS los archivos de EnRevision/ (incluye _DUDOSOS),
            # indexados por su nombre real: es el que conocen los Excel y el que
            # tiene que quedar en Historico/.
            procesado_files = {}  # segmento → (clase, path_src)
            if os.path.exists(rt.ruta_procesado):
                for entry in os.scandir(rt.ruta_procesado):
                    if not entry.is_dir():
                        continue
                    if entry.name == "_DUDOSOS":
                        for clase_entry in os.scandir(entry.path):
                            if not clase_entry.is_dir():
                                continue
                            for f in os.scandir(clase_entry.path):
                                if f.is_file():
                                    nombre = sin_confianza(f.name, nombres_reales)
                                    procesado_files[nombre] = (clase_entry.name, f.path)
                    else:
                        for f in os.scandir(entry.path):
                            if f.is_file():
                                nombre = sin_confianza(f.name, nombres_reales)
                                procesado_files[nombre] = (entry.name, f.path)

            # ---- Parte 1: asentar EnRevision/ → Historico/ (si hay algo) ----
            if procesado_files:
                names = set(procesado_files.keys())

                # Eliminar versiones viejas de esas imágenes en Historico/
                if os.path.exists(rt.ruta_salida):
                    for root_dir, dirs, files in os.walk(rt.ruta_salida):
                        for fname in files:
                            if fname in names:
                                os.remove(os.path.join(root_dir, fname))

                # Copiar todo a Historico/CLASE/ (plano, sin _DUDOSOS).
                # Enlace, no copia: son los mismos bytes que ya están en
                # Segmentos/ y en EnRevision/, y en un rato se borra EnRevision/
                # entero — borrar un enlace no toca a los otros (ver core.files).
                for fname, (clase, src) in procesado_files.items():
                    dst_dir = os.path.join(rt.ruta_salida, clase)
                    os.makedirs(dst_dir, exist_ok=True)
                    enlazar_o_copiar(src, os.path.join(dst_dir, fname))

                # Sincronizar class_pred con la ubicación física real en
                # EnRevision/: si el usuario movió una imagen a otra carpeta de
                # clase, procesado_files ya refleja esa carpeta nueva pero
                # df_proc todavía tiene la clase vieja de EnRevision.xlsx.
                #
                # Cuando la clase cambió es una reclasificación manual y la
                # decisión humana es definitiva: además de seguir a la carpeta
                # nueva se fuerza confidence = 1.0 (mismo criterio que
                # actualizar_clasificados). Sin esto la imagen queda asentada
                # con la confianza vieja del modelo y el filtrar_dudosos() de
                # más abajo la manda a _DUDOSOS pese a estar clasificada a mano.
                #
                # Mover una imagen dentro de la MISMA clase (sacarla de _DUDOSOS
                # sin cambiarle la etiqueta) no cuenta como señal acá: el umbral
                # con el que se asienta puede no ser el usado al clasificar
                # (para eso está "Refiltrar").
                if df_proc is not None:
                    seg_col = df_proc.columns[0]
                    for i, row in df_proc.iterrows():
                        nombre = str(row[seg_col])
                        if nombre not in procesado_files:
                            continue
                        clase_carpeta = procesado_files[nombre][0]
                        if str(row["class_pred"]) != clase_carpeta:
                            df_proc.at[i, "class_pred"] = clase_carpeta
                            df_proc.at[i, "confidence"] = 1.0

                # Merge EnRevision.xlsx → Historico.xlsx
                clas_excel = rt.data_path("Datos", "Historico.xlsx")
                if df_proc is not None:
                    proc_names = nombres_reales
                    if os.path.exists(clas_excel):
                        df_clas = pd.read_excel(clas_excel)
                        df_clas = normalize_prediction_columns(df_clas)
                        df_clas = df_clas[~df_clas.iloc[:, 0].astype(str).isin(proc_names)]
                        df_merged = pd.concat([df_clas, df_proc], ignore_index=True)
                    else:
                        df_merged = df_proc
                    os.makedirs(os.path.dirname(clas_excel), exist_ok=True)
                    df_merged.to_excel(clas_excel, index=False)

                # Aplicar split seguros/dudosos sobre Historico/
                filtrar_dudosos(self.umbral)

                # Limpiar EnRevision/
                shutil.rmtree(rt.ruta_procesado, ignore_errors=True)
                os.makedirs(rt.ruta_procesado, exist_ok=True)
                if os.path.exists(rt.procesado_excel):
                    os.remove(rt.procesado_excel)

                self.log.emit(f"✅ {len(procesado_files)} imágenes asentadas en Historico/")
            else:
                self.log.emit("ℹ️ EnRevision/ vacío: nada para asentar.")

            # ---- Parte 2: reconciliar Historico/ con sus Excel ----
            # Antes era el botón "Actualizar". Barato y casi siempre un no-op
            # justo después de asentar; agarra los cambios que el usuario haya
            # hecho a mano directamente en Historico/.
            self.log.emit("--- reconciliando Historico/ con los Excel ---")
            actualizar_clasificados(
                clasificados_excel=rt.data_path("Datos", "Historico.xlsx"),
                seguros_excel=rt.data_path("Datos", "Seguros.xlsx"),
                dudosos_excel=rt.data_path("Datos", "Dudosos.xlsx"),
                clasificados_dir=rt.ruta_salida,
            )
            self.log.emit("✔ Listo.")

        except Exception as e:
            self.log.emit(f"❌ Error al asentar: {e}")
        self.finished.emit()
