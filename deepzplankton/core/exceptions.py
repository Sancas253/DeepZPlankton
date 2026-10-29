class ProcesoCancelado(Exception):
    """El usuario canceló una segmentación o clasificación en curso.

    Distinta de una excepción real para que quien la atrape (el hilo de la
    GUI) pueda loguear "cancelado por el usuario" en vez de "❌ Error".
    """
