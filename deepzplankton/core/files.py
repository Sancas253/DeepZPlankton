"""
El objetivo de este archivo es manejar las imagenes que se cuadruplican en el proceso para que sean
hard links, lo que logra la funcionalidad esperada, que es que las imagenes vivan en 4 carpetas distintas
pero sin el coste de que esto sea así
"""
import os
import shutil


def liberar_destino(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def enlazar_o_copiar(src, dst):
    liberar_destino(dst)
    try:
        os.link(src, dst)
    except (OSError, AttributeError, NotImplementedError):
        shutil.copy2(src, dst)


def vaciar_carpeta(ruta):
    """Borra todo el contenido de ruta, sin borrar la carpeta en sí."""
    for item in os.listdir(ruta):
        item_path = os.path.join(ruta, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
