"""Resumen en texto y conteo de lo que hay pendiente en EnRevision/."""
import os

from . import runtime_paths as rt


def resumen_procesado(umbral):
    """Devuelve string con resumen de EnRevision.xlsx según umbral."""
    try:
        import pandas as pd
        if not os.path.exists(rt.procesado_excel):
            return ""
        df = pd.read_excel(rt.procesado_excel)
        if df.empty or "class_pred" not in df.columns or "confidence" not in df.columns:
            return ""
        total = len(df)
        n_seg = int((df["confidence"] >= umbral).sum())
        n_dud = total - n_seg
        counts = df.groupby("class_pred")["confidence"].agg(
            total="count",
            seguros=lambda x: (x >= umbral).sum(),
        ).sort_values("total", ascending=False)
        lines = [
            f"\n📊 Resumen (umbral={umbral}):",
            f"   Total: {total}   ✅ Seguros: {n_seg}   ⚠️ Dudosos: {n_dud}\n",
        ]
        for clase, row in counts.iterrows():
            dud = int(row.total) - int(row.seguros)
            lines.append(f"   {clase}: {int(row.total)}  (S:{int(row.seguros)}  D:{dud})")
        return "\n".join(lines)
    except Exception:
        return ""


def count_procesado():
    """Cuenta archivos en EnRevision/ (incluyendo _DUDOSOS)."""
    if not os.path.isdir(rt.ruta_procesado):
        return 0
    total = 0
    for entry in os.scandir(rt.ruta_procesado):
        if not entry.is_dir():
            continue
        if entry.name == "_DUDOSOS":
            for sub in os.scandir(entry.path):
                if sub.is_dir():
                    total += sum(1 for f in os.scandir(sub.path) if f.is_file())
        else:
            total += sum(1 for f in os.scandir(entry.path) if f.is_file())
    return total
