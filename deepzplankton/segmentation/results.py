"""Excel acumulativo de mediciones, compartido por los dos segmentadores."""
import os

import pandas as pd


def guardar_resultados(excel_path, all_results):
    """Escribe all_results en excel_path, reemplazando cualquier fila previa
    con el mismo 'segment' (re-segmentar reusa nombres de archivo)."""
    if all_results:
        df_new = pd.DataFrame(all_results)
        if os.path.exists(excel_path):
            df_existing = pd.read_excel(excel_path)
            new_names = set(df_new["segment"].astype(str))
            df_existing = df_existing[~df_existing["segment"].astype(str).isin(new_names)]
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final.to_excel(excel_path, index=False)
        else:
            df_new.to_excel(excel_path, index=False)
    elif not os.path.exists(excel_path):
        pd.DataFrame().to_excel(excel_path, index=False)
