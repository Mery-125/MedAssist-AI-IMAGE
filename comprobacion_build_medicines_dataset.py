import pandas as pd

'''
df = pd.read_csv("dataset_medicamentos.csv")

conteo = (
    df["macroclase_15_nombre"]
    .fillna("None")
    .value_counts()
    .rename_axis("macroclase_15_nombre")
    .reset_index(name="num_imagenes")
)

print(conteo)
conteo.to_csv("num_imagenes_por_macroclase.csv", index=False, encoding="utf-8")


'''

df = pd.read_csv("dataset_medicamentos.csv")

resumen = (
    df.groupby("macroclase_15_nombre")
      .agg(
          num_imagenes=("sample_id", "count"),
          num_medicamentos=("nregistro", "nunique")
      )
      .reset_index()
      .sort_values("num_medicamentos", ascending=False)
)

print(resumen)
resumen.to_csv("resumen_imagenes_y_medicamentos_por_clase.csv", index=False, encoding="utf-8")