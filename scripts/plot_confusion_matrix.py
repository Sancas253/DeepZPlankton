"""
Visualiza la matriz de confusión guardada en cv_results.json o meta.json.

Uso en Colab:
    !python plot_confusion_matrix.py --json /content/swin_tiny_finetuned/cv_results.json
    !python plot_confusion_matrix.py --json /content/swin_tiny_finetuned/fold_0/meta.json
    !python plot_confusion_matrix.py --json cv_results.json --normalize --output cm.png
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_cm_and_classes(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # cv_results.json  → pooled OOF
    if "pooled_oof" in data:
        cm = np.array(data["pooled_oof"]["confusion_matrix"])
        # class_names vive en fold_0/meta.json; intentamos cargarlo desde la misma carpeta
        meta_path = os.path.join(os.path.dirname(json_path), "fold_0", "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f2:
                class_names = json.load(f2)["class_names"]
        else:
            class_names = [str(i) for i in range(cm.shape[0])]
        title = "Matriz de confusión — OOF global (todos los folds)"

    # meta.json de un fold individual
    elif "confusion_matrix" in data:
        cm = np.array(data["confusion_matrix"])
        class_names = data.get("class_names", [str(i) for i in range(cm.shape[0])])
        fold = data.get("fold", "?")
        title = f"Matriz de confusión — Fold {fold}"

    else:
        raise ValueError("No se encontró 'confusion_matrix' ni 'pooled_oof' en el JSON.")

    return cm, class_names, title


def plot_cm(cm, class_names, title, normalize, output):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums == 0, 0, cm / row_sums)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = cm.max()

    n = len(class_names)
    fig_size = max(14, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        linecolor="lightgray",
        vmin=0,
        vmax=vmax,
        ax=ax,
        annot_kws={"size": max(5, 9 - n // 10)},
    )

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Predicho", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=max(5, 8 - n // 15))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=max(5, 8 - n // 15))

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"✅ Guardado en: {output}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",      required=True, help="Ruta al cv_results.json o meta.json")
    parser.add_argument("--normalize", action="store_true", help="Normalizar por fila (recall por clase)")
    parser.add_argument("--output",    default="confusion_matrix.png", help="Archivo de salida")
    args = parser.parse_args()

    cm, class_names, title = load_cm_and_classes(args.json)
    plot_cm(cm, class_names, title, args.normalize, args.output)


if __name__ == "__main__":
    main()
