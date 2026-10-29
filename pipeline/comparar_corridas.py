"""
comparar_corridas.py — compara varios modelos imagen por imagen (oof_predicciones.csv).

Para demostrar si el techo de precisión está en los datos y no en los modelos:
  1. McNemar exacto entre cada par de modelos: ¿la diferencia es significativa?
  2. Superposición de errores: cuántas imágenes fallan los dos, contra las que se
     esperarían si cada modelo fallara al azar (dentro de cada clase, así no
     cuenta solo que algunas clases son más difíciles).
  3. Distribución de "cuántos modelos fallan cada imagen", observada contra la
     esperada si los errores fueran independientes. Si muchos más fallan en
     TODOS los modelos que lo esperado, esas imágenes son difíciles en sí
     (ambiguas o mal etiquetadas), no debilidades de un modelo.
  4. dificiles.csv: las imágenes que fallan (casi) todos, listas para revisar a
     mano; con --dataset se copian a carpetas real__como__predicha/.

Uso:
  python comparar_corridas.py DINOv2FT=corridas/dinov2_ft  Fusion=corridas/convnext_dinov2_fusion_plus \
      ConvNeXtPlus=corridas/convnext_tiny_plus/oof_predicciones.csv --salida comparacion \
      --dataset /content/deepz/dataset_limpio
Cada argumento es NOMBRE=carpeta del entrenamiento (con oof_predicciones.csv) o el .csv.
"""
import argparse
import csv
import itertools
import os
import shutil

import numpy as np

from predicciones import leer_oof


def mcnemar_exacto(b, c):
    """p bilateral exacto: b = solo A acierta, c = solo B acierta."""
    from scipy.stats import binomtest
    return 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue


def holm(pvalores):
    orden = np.argsort(pvalores)
    m, ajust, previo = len(pvalores), np.empty(len(pvalores)), 0.0
    for k, i in enumerate(orden):
        previo = max(previo, min(1.0, (m - k) * pvalores[i]))
        ajust[i] = previo
    return ajust


def poisson_binomial(probs):
    """Distribución de la cantidad de éxitos con probabilidades distintas (DP)."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for p in probs:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return dist


def cargar(corridas):
    nombres, datos = [], []
    for arg in corridas:
        if "=" not in arg:
            raise SystemExit(f"Formato NOMBRE=ruta: {arg}")
        nombre, ruta = arg.split("=", 1)
        nombres.append(nombre)
        datos.append(leer_oof(ruta))
    comunes = sorted(set.intersection(*(set(d) for d in datos)))
    for n, d in zip(nombres, datos):
        if len(d) != len(comunes):
            print(f"⚠️  {n}: {len(d) - len(comunes)} imágenes que no están en todos los modelos (se ignoran)")
    if not comunes:
        raise SystemExit("No hay imágenes en común: ¿mismo dataset?")
    reales = [datos[0][r]["real"] for r in comunes]
    for n, d in zip(nombres, datos):
        distintas = sum(d[r]["real"] != real for r, real in zip(comunes, reales))
        if distintas:
            raise SystemExit(f"{n}: {distintas} imágenes con otra etiqueta real: ¿otro mapeo de clases?")
    return nombres, datos, comunes, reales


def main(argv=None):
    p = argparse.ArgumentParser(description="McNemar y errores compartidos entre modelos.")
    p.add_argument("corridas", nargs="+", help="NOMBRE=carpeta_o_csv (al menos 2)")
    p.add_argument("--salida", default="comparacion_corridas")
    p.add_argument("--min-fallan", type=int, default=0,
                   help="Imágenes difíciles = las que fallan al menos N modelos (0 = todos)")
    p.add_argument("--dataset", help="Carpeta del dataset: copia las imágenes difíciles para revisarlas")
    args = p.parse_args(argv)
    if len(args.corridas) < 2:
        raise SystemExit("Hacen falta al menos 2 modelos.")

    nombres, datos, rutas, reales = cargar(args.corridas)
    M, N = len(nombres), len(rutas)
    acierto = np.array([[int(d[r]["acierto"]) for d in datos] for r in rutas], dtype=bool)
    pred = [[d[r]["predicha"] for d in datos] for r in rutas]
    conf = np.array([[float(d[r]["confianza"]) for d in datos] for r in rutas])
    reales = np.array(reales)
    os.makedirs(args.salida, exist_ok=True)
    lineas = [f"# Comparación de {M} modelos sobre {N} imágenes (predicciones out-of-fold)\n"]

    # ---- Precisión por modelo ----
    lineas.append("| Modelo | Accuracy | Errores |\n|---|---|---|")
    for j, n in enumerate(nombres):
        lineas.append(f"| {n} | {acierto[:, j].mean():.4f} | {(~acierto[:, j]).sum()} |")

    # ---- McNemar + superposición por par ----
    # Error esperado por azar dentro de cada clase: p_A(clase) * p_B(clase) * n_clase
    clases = sorted(set(reales))
    tasa = {c: (~acierto[reales == c]).mean(0) for c in clases}
    filas, pv = [], []
    for a, b in itertools.combinations(range(M), 2):
        solo_a = int((acierto[:, a] & ~acierto[:, b]).sum())
        solo_b = int((~acierto[:, a] & acierto[:, b]).sum())
        ambos_mal = int((~acierto[:, a] & ~acierto[:, b]).sum())
        esperado = sum(tasa[c][a] * tasa[c][b] * (reales == c).sum() for c in clases)
        union = int((~acierto[:, a] | ~acierto[:, b]).sum())
        pv.append(mcnemar_exacto(solo_a, solo_b))
        filas.append([nombres[a], nombres[b], solo_a, solo_b, ambos_mal, esperado,
                      ambos_mal / max(esperado, 1e-9), ambos_mal / max(union, 1)])
    p_holm = holm(np.array(pv))
    with open(os.path.join(args.salida, "pares.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["A", "B", "solo_A_acierta", "solo_B_acierta", "ambos_fallan",
                    "ambos_fallan_esperado_azar", "veces_sobre_azar", "jaccard_errores",
                    "p_mcnemar", "p_holm"])
        for fila, p_, ph in zip(filas, pv, p_holm):
            w.writerow(fila[:5] + [f"{fila[5]:.1f}", f"{fila[6]:.1f}", f"{fila[7]:.3f}",
                                   f"{p_:.3g}", f"{ph:.3g}"])

    lineas.append("\n## McNemar exacto por par\n")
    lineas.append("Solo A / solo B = imágenes que acierta un modelo y el otro no. "
                  "p Holm corrige por hacer varias comparaciones.\n")
    lineas.append("| A | B | solo A | solo B | p | p Holm | ¿Diferencia? |\n|---|---|---|---|---|---|---|")
    for fila, p_, ph in zip(filas, pv, p_holm):
        mejor = fila[0] if fila[2] > fila[3] else fila[1]
        veredicto = f"sí, mejor {mejor}" if ph < 0.05 else "no significativa"
        lineas.append(f"| {fila[0]} | {fila[1]} | {fila[2]} | {fila[3]} | {p_:.3g} | {ph:.3g} | {veredicto} |")

    lineas.append("\n## Errores compartidos\n")
    lineas.append("Esperado por azar = si cada modelo fallara al azar con su tasa de error "
                  "de cada clase. Muchas más veces que el azar = los modelos fallan en las "
                  "mismas imágenes.\n")
    lineas.append("| A | B | fallan ambos | esperado por azar | veces sobre azar | Jaccard |\n|---|---|---|---|---|---|")
    for fila in filas:
        lineas.append(f"| {fila[0]} | {fila[1]} | {fila[4]} | {fila[5]:.1f} | {fila[6]:.1f}× | {fila[7]:.2f} |")

    # ---- ¿Cuántos modelos fallan cada imagen? observado vs independiente ----
    k_obs = (~acierto).sum(1)
    observado = np.bincount(k_obs, minlength=M + 1)
    esperado = np.zeros(M + 1)
    for c in clases:
        esperado += poisson_binomial(tasa[c]) * (reales == c).sum()
    lineas.append("\n## Cuántos modelos fallan cada imagen\n")
    lineas.append("Esperado = si los errores fueran independientes entre modelos (por clase).\n")
    lineas.append("| Modelos que fallan | Imágenes observadas | Esperadas si independientes |\n|---|---|---|")
    for k in range(M + 1):
        lineas.append(f"| {k} | {observado[k]} | {esperado[k]:.1f} |")
    todos = observado[M]
    lineas.append(f"\n**{todos} imágenes fallan en los {M} modelos; si los errores fueran "
                  f"independientes se esperarían {esperado[M]:.1f}** "
                  f"({todos / max(esperado[M], 1e-9):.0f}× más).")
    lineas.append(f"Los modelos se equivocan en {(k_obs > 0).sum()} imágenes distintas; "
                  f"{todos} ({todos / max((k_obs > 0).sum(), 1):.0%}) son errores de todos.")

    # ---- Imágenes difíciles ----
    minimo = args.min_fallan or M
    dificiles = [i for i in np.argsort(-k_obs, kind="stable") if k_obs[i] >= minimo]
    with open(os.path.join(args.salida, "dificiles.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ruta", "real", "modelos_que_fallan"] +
                   [f"pred_{n}" for n in nombres] + [f"conf_{n}" for n in nombres])
        for i in dificiles:
            w.writerow([rutas[i], reales[i], k_obs[i]] + pred[i] + [f"{x:.3f}" for x in conf[i]])
    pares = {}
    for i in dificiles:
        mal = [x for x, ok in zip(pred[i], acierto[i]) if not ok]
        destino = max(set(mal), key=mal.count)
        pares[(reales[i], destino)] = pares.get((reales[i], destino), 0) + 1
    lineas.append(f"\n## Imágenes que fallan al menos {minimo} modelos: {len(dificiles)}\n")
    lineas.append("| Real | Predicha (la más votada) | Imágenes |\n|---|---|---|")
    for (r, d), n in sorted(pares.items(), key=lambda t: -t[1])[:20]:
        lineas.append(f"| {r} | {d} | {n} |")
    lineas.append("\nListado completo en dificiles.csv.")

    if args.dataset:
        base = os.path.join(args.salida, "dificiles")
        for i in dificiles:
            mal = [x for x, ok in zip(pred[i], acierto[i]) if not ok]
            sub = f"{reales[i]}__como__{max(set(mal), key=mal.count)}"
            origen = os.path.join(args.dataset, rutas[i])
            if os.path.exists(origen):
                os.makedirs(os.path.join(base, sub), exist_ok=True)
                shutil.copy2(origen, os.path.join(base, sub, os.path.basename(rutas[i])))
        lineas.append(f"Imágenes copiadas a {base}/<real>__como__<predicha>/ para revisar.")

    graficar(nombres, filas, p_holm, observado, esperado, args.salida)
    texto = "\n".join(lineas)
    with open(os.path.join(args.salida, "resumen.md"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(texto)
    print(f"\n✅ Resultados en {os.path.abspath(args.salida)}")


def graficar(nombres, filas, p_holm, observado, esperado, salida):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    M = len(nombres)
    idx = {n: i for i, n in enumerate(nombres)}
    veces = np.full((M, M), np.nan)
    pmat = np.full((M, M), np.nan)
    for fila, ph in zip(filas, p_holm):
        a, b = idx[fila[0]], idx[fila[1]]
        veces[a, b] = veces[b, a] = fila[6]
        pmat[a, b] = pmat[b, a] = ph
    fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 5.5))
    for ax, mat, titulo, fmt, cmap in ((axes[0], veces, "Errores compartidos\n(veces sobre lo esperado por azar)", "{:.1f}×", "Reds"),
                                       (axes[1], pmat, "McNemar (p corregido por Holm)\nverde = diferencia significativa", "{:.2g}", "RdYlGn_r")):
        im = ax.imshow(mat if cmap == "Reds" else np.log10(np.clip(mat, 1e-12, 1)), cmap=cmap)
        ax.set_xticks(range(M), nombres, rotation=45, ha="right")
        ax.set_yticks(range(M), nombres)
        for i in range(M):
            for j in range(M):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=9)
        ax.set_title(titulo)
    k = np.arange(M + 1)
    axes[2].bar(k - 0.2, observado, 0.4, label="observado")
    axes[2].bar(k + 0.2, esperado, 0.4, label="si los errores fueran independientes")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("cantidad de modelos que fallan la imagen")
    axes[2].set_ylabel("imágenes (escala log)")
    axes[2].set_xticks(k)
    axes[2].legend()
    axes[2].set_title("¿Los modelos fallan en las mismas imágenes?")
    fig.tight_layout()
    fig.savefig(os.path.join(salida, "comparacion.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
