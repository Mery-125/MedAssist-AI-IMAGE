import json
from collections import Counter, defaultdict
from pathlib import Path
import csv

JSON_PATH = "medicamentos_5000_con_texto.json"   # cambia esta ruta
OUT_DIR = "resumen_atc"

ATC1_MAP = {
    "A": "Tracto alimentario y metabolismo",
    "B": "Sangre y órganos hematopoyéticos",
    "C": "Sistema cardiovascular",
    "D": "Dermatológicos",
    "G": "Sistema genitourinario y hormonas sexuales",
    "H": "Preparados hormonales sistémicos, excl. hormonas sexuales e insulinas",
    "J": "Antiinfecciosos para uso sistémico",
    "L": "Antineoplásicos e inmunomoduladores",
    "M": "Sistema musculoesquelético",
    "N": "Sistema nervioso",
    "P": "Antiparasitarios, insecticidas y repelentes",
    "R": "Sistema respiratorio",
    "S": "Órganos de los sentidos",
    "V": "Varios",
}


def load_records(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("results", "medicamentos", "items", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("No se encontró una lista de medicamentos en el JSON.")


def get_best_atc_code(med):
    """
    Coge el ATC más específico disponible:
    preferencia nivel 5 > 4 > 3 > 2.
    """
    atcs = med.get("atcs", [])
    if not isinstance(atcs, list):
        return None, None, None

    best = None
    best_level = -1

    for atc in atcs:
        if not isinstance(atc, dict):
            continue
        nivel = atc.get("nivel")
        codigo = atc.get("codigo")
        nombre = atc.get("nombre")
        if codigo and isinstance(nivel, int) and nivel > best_level:
            best = (codigo, nombre, nivel)
            best_level = nivel

    return best if best else (None, None, None)


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(JSON_PATH)

    atc1_counter = Counter()
    atc2_counter = Counter()
    atc3_counter = Counter()

    atc2_examples = defaultdict(list)
    atc3_examples = defaultdict(list)

    rows = []

    for med in records:
        nombre_med = med.get("nombre")
        codigo, nombre_atc, nivel = get_best_atc_code(med)

        if not codigo:
            continue

        codigo = str(codigo).strip().upper()

        # Derivados
        atc1 = codigo[0] if len(codigo) >= 1 else None
        atc2 = codigo[:3] if len(codigo) >= 3 else None
        atc3 = codigo[:4] if len(codigo) >= 4 else None

        atc1_nombre = ATC1_MAP.get(atc1, "Desconocido")

        atc1_counter[(atc1, atc1_nombre)] += 1
        if atc2:
            atc2_counter[atc2] += 1
            if nombre_med and len(atc2_examples[atc2]) < 5:
                atc2_examples[atc2].append(nombre_med)

        if atc3:
            atc3_counter[atc3] += 1
            if nombre_med and len(atc3_examples[atc3]) < 5:
                atc3_examples[atc3].append(nombre_med)

        rows.append({
            "nombre_medicamento": nombre_med,
            "codigo_atc_original": codigo,
            "nivel_atc_original": nivel,
            "atc1_codigo": atc1,
            "atc1_nombre": atc1_nombre,
            "atc2_codigo": atc2,
            "atc3_codigo": atc3,
        })

    # -----------------------------
    # Mostrar por pantalla
    # -----------------------------
    print("=" * 70)
    print("RESUMEN RECOMENDADO PARA REDUCIR CLASES")
    print("=" * 70)
    print(f"Medicamentos procesados: {len(rows)}")
    print(f"Nº grupos ATC1 (muy grueso): {len(atc1_counter)}")
    print(f"Nº grupos ATC2 (intermedio): {len(atc2_counter)}")
    print(f"Nº grupos ATC3 (más fino): {len(atc3_counter)}")
    print()

    print("=" * 70)
    print("ATC1 = GRUPO PRINCIPAL (MUY RECOMENDABLE PARA TU PROYECTO)")
    print("=" * 70)
    for (codigo, nombre), n in atc1_counter.most_common():
        print(f"{codigo} -> {nombre}: {n} medicamentos")
    print()

    print("=" * 70)
    print("TOP 30 ATC2")
    print("=" * 70)
    for codigo, n in atc2_counter.most_common(30):
        print(f"{codigo}: {n} medicamentos | ejemplos: {atc2_examples[codigo]}")
    print()

    print("=" * 70)
    print("TOP 30 ATC3")
    print("=" * 70)
    for codigo, n in atc3_counter.most_common(30):
        print(f"{codigo}: {n} medicamentos | ejemplos: {atc3_examples[codigo]}")
    print()

    # -----------------------------
    # Guardar CSVs
    # -----------------------------
    # Dataset base con columnas derivadas
    with open(out_dir / "dataset_atc_reducido.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "nombre_medicamento",
                "codigo_atc_original",
                "nivel_atc_original",
                "atc1_codigo",
                "atc1_nombre",
                "atc2_codigo",
                "atc3_codigo",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Resumen ATC1
    with open(out_dir / "resumen_atc1.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["atc1_codigo", "atc1_nombre", "num_medicamentos"])
        for (codigo, nombre), n in atc1_counter.most_common():
            writer.writerow([codigo, nombre, n])

    # Resumen ATC2
    with open(out_dir / "resumen_atc2.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["atc2_codigo", "num_medicamentos", "ejemplo_1", "ejemplo_2", "ejemplo_3", "ejemplo_4", "ejemplo_5"])
        for codigo, n in atc2_counter.most_common():
            ejemplos = atc2_examples[codigo] + [""] * (5 - len(atc2_examples[codigo]))
            writer.writerow([codigo, n] + ejemplos[:5])

    # Resumen ATC3
    with open(out_dir / "resumen_atc3.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["atc3_codigo", "num_medicamentos", "ejemplo_1", "ejemplo_2", "ejemplo_3", "ejemplo_4", "ejemplo_5"])
        for codigo, n in atc3_counter.most_common():
            ejemplos = atc3_examples[codigo] + [""] * (5 - len(atc3_examples[codigo]))
            writer.writerow([codigo, n] + ejemplos[:5])

    print(f"Archivos guardados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()