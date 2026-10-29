"""Rutas de trabajo de la app en tiempo de ejecución.

Se calculan una sola vez al importar el módulo. Los hilos y widgets que las
necesitan importan este módulo (no los nombres sueltos) y acceden vía
`runtime_paths.X`, para que un test pueda parchear una ruta con
monkeypatch.setattr(runtime_paths, "X", ...) y que todo el que la lea después
del parche vea el valor nuevo.
"""
import os

from ..core.paths import app_path, data_path, is_model_dir  # noqa: F401  (re-exportadas)

BASE_DIR = app_path()
# Los modelos que agrega el usuario van junto al .exe, no dentro de _internal:
# esa carpeta la regenera PyInstaller en cada build, así que un modelo instalado
# ahí desaparece en la próxima actualización (y bajo Program Files ni siquiera
# se puede escribir). MODEL_BUNDLE sigue siendo el modelo que viene con la app.
MODEL_BASE   = data_path("model")
MODEL_BUNDLE = app_path("model")

ruta_test       = data_path("Segmentos", "_Cache")
# EnRevision/: la tanda que se acaba de correr, todavía sin confirmar (ahí se
# revisa a mano). Historico/: todo lo ya confirmado con "Asentar", acumulado.
# Se llamaban "Procesado" y "Clasificados"; ver _migrar_nombres_viejos.
ruta_procesado  = data_path("EnRevision")
ruta_salida     = data_path("Historico")
excel_path      = data_path("Datos", "Mediciones.xlsx")
procesado_excel = data_path("Datos", "EnRevision.xlsx")
clasificados_excel = data_path("Datos", "Historico.xlsx")


def migrar_nombres_viejos():
    """Si existe una instalación previa con las carpetas/Excel viejos
    (Procesado/Clasificados), los renombra a los nuevos para no perder el
    trabajo ya hecho. No pisa nada: si el nombre nuevo ya existe, no toca
    el viejo.

    Se llama explícitamente desde main() al arrancar la app — no acá arriba
    al nivel de módulo, porque este módulo también lo importan los tests
    (con data_path parcheado a un tmp_path) y un rename automático en el
    import tocaría el disco real antes de que el test pueda pisarlo."""
    pares = [
        (data_path("Procesado"), ruta_procesado),
        (data_path("Clasificados"), ruta_salida),
        (data_path("Datos", "Procesado.xlsx"), procesado_excel),
        (data_path("Datos", "Clasificados.xlsx"), clasificados_excel),
    ]
    for viejo, nuevo in pares:
        if os.path.exists(viejo) and not os.path.exists(nuevo):
            os.rename(viejo, nuevo)
