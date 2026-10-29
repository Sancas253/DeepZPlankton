"""Prefijo de confianza — sólo para los archivos de EnRevision/

Revisar quiere decir mirar primero lo que el modelo menos tuvo claro, y eso
se hace a mano, no acá adentro. Anteponiendo la confianza en
el nombre ("048_EH0222_12.jpg"), ordenar por nombre ya es ordenar por duda:
los tres dígitos con ceros a la izquierda hacen que el orden alfabético y el
numérico coincidan.
"""
import re

_PREFIJO_CONFIANZA = re.compile(r"^\d{3}_")


def con_confianza(nombre, confianza, nombres_reales=None):
    """'EH0222_12.jpg' + 0.4812 -> '048_EH0222_12.jpg'.

    Limpia primero para no apilar prefijos al reclasificar, con la misma
    salvedad que sin_confianza(): si el nombre verdadero ya empieza con tres
    dígitos, el conjunto de nombres reales es lo único que lo salva.
    """
    pct = int(round(confianza * 100))
    # "reviso todo lo que no diga 100" es un criterio de revisión razonable, así
    # que 100 tiene que significar 100: un 0.9958 redondeado para arriba se
    # saltearía la revisión sin que nadie se entere. Se redondea normal salvo
    # ese caso. (round() y no truncar, porque 0.61*100 da 60.999… en binario.)
    if pct >= 100 and confianza < 1.0:
        pct = 99
    return f"{pct:03d}_{sin_confianza(nombre, nombres_reales)}"


def sin_confianza(nombre, nombres_reales=None):
    """Devuelve el nombre real del segmento, con o sin prefijo puesto.

    Mirando el nombre solo no siempre se puede decidir: si la imagen de campaña
    se llamaba "001_muestra.tif", el segmento se llama "001_muestra_3.jpg" y el
    prefijo es parte del nombre. Por eso quien tenga a mano el conjunto de
    nombres verdaderos (las predicciones, o la columna del Excel) lo pasa y ese
    conjunto manda.
    """
    if nombres_reales is not None and nombre in nombres_reales:
        return nombre
    return _PREFIJO_CONFIANZA.sub("", nombre, count=1)
