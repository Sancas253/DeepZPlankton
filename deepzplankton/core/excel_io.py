"""Helpers de lectura/escritura de las planillas de predicciones."""
import os

import pandas as pd


def upsert_excel_by_key(path, df_new, key_col="segment"):
    """Agrega df_new a path, reemplazando filas que compartan la misma clave."""
    if df_new.empty:
        return
    if os.path.exists(path):
        df_prev = pd.read_excel(path)
        if not df_prev.empty:
            key = key_col if key_col in df_prev.columns else df_prev.columns[0]
            new_keys = set(df_new[key_col].astype(str))
            df_prev = df_prev[~df_prev[key].astype(str).isin(new_keys)]
        df_out = pd.concat([df_prev, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_excel(path, index=False)


def append_to_excel(path, df, sheet_name="Sheet1"):
    if not os.path.exists(path):
        df.to_excel(path, index=False, sheet_name=sheet_name)
        return
    with pd.ExcelWriter(path, engine="openpyxl", mode="a",
                        if_sheet_exists="overlay") as writer:
        startrow = writer.book[sheet_name].max_row if sheet_name in writer.book.sheetnames else 0
        df.to_excel(
            writer, sheet_name=sheet_name, index=False,
            header=(startrow == 0), startrow=startrow,
        )


def normalize_prediction_columns(df):
    df.columns = [c.strip() for c in df.columns]
    if "class_pred" not in df.columns:
        df["class_pred"] = ""
    else:
        df["class_pred"] = df["class_pred"].fillna("").astype(str)
    if "confidence" not in df.columns:
        df["confidence"] = 0.0
    else:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    return df


def merge_predictions_into_excel(pred_dict, excel_path, output_excel_path=None):
    """Vuelca pred_dict (segmento -> (clase, confianza)) sobre Mediciones.xlsx
    y guarda el resultado en output_excel_path (por defecto Historico.xlsx
    junto a excel_path).

    Si excel_path no existe (se saltó la segmentación), genera un
    Historico.xlsx mínimo sólo con las predicciones.
    """
    new_names = set(pred_dict.keys())

    if not os.path.exists(excel_path):
        print(f"⚠ No se encontró {os.path.basename(excel_path)}. "
              "Generando Historico.xlsx solo con predicciones.")
        df_new = pd.DataFrame([
            {"segment": name, "class_pred": cls, "confidence": round(conf, 6)}
            for name, (cls, conf) in pred_dict.items()
        ])
    else:
        df = pd.read_excel(excel_path)
        df = normalize_prediction_columns(df)
        seg_col = "segment" if "segment" in df.columns else df.columns[0]
        if seg_col != "segment":
            print(f"⚠ Columna 'segment' no encontrada en Mediciones; usando '{seg_col}'.")

        for i, row in df.iterrows():
            nombre = str(row[seg_col]).strip()
            if nombre in pred_dict:
                clase, conf = pred_dict[nombre]
                df.at[i, "class_pred"] = clase
                df.at[i, "confidence"] = round(conf, 6)

        df_known = df[df[seg_col].isin(new_names)].copy()
        df_known = df_known.drop_duplicates(subset=seg_col)

        # Images absent from Mediciones (direct classification without prior segmentation)
        known_names = set(df_known[seg_col].astype(str))
        missing = new_names - known_names
        if missing:
            print(f"⚠ {len(missing)} segmento(s) no están en Mediciones; "
                  "agregando solo con predicción.")
            df_missing = pd.DataFrame([
                {"segment": name, "class_pred": cls, "confidence": round(conf, 6)}
                for name, (cls, conf) in pred_dict.items()
                if name in missing
            ])
            df_new = pd.concat([df_known, df_missing], ignore_index=True)
        else:
            df_new = df_known

    output_excel = output_excel_path or os.path.join(os.path.dirname(excel_path), "Historico.xlsx")
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    if os.path.exists(output_excel):
        df_prev = pd.read_excel(output_excel)
        df_prev = normalize_prediction_columns(df_prev)
        prev_col = "segment" if "segment" in df_prev.columns else df_prev.columns[0]
        df_prev = df_prev[~df_prev[prev_col].isin(new_names)]
        pd.concat([df_prev, df_new], ignore_index=True).to_excel(output_excel, index=False)
    else:
        df_new.to_excel(output_excel, index=False)

    print(f"✅ Excel guardado en:\n{output_excel}")
    return output_excel
